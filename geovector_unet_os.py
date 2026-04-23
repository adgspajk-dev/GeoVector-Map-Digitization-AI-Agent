"""
================================================================================
GEOVECTOR OS — Open-Source Geological Map Digitization + PyTorch U-Net
================================================================================
Architecture: LLM-assisted Legend Parsing → U-Net Texture Segmentation → GeoPandas 

DEPENDENCIES:
  - torch, torchvision
  - opencv-python (cv2)
  - numpy, scipy, scikit-image
  - anthropic (Claude API)
  - geopandas, rasterio, shapely, matplotlib

USAGE:
  python geovector_unet_os.py
================================================================================
"""

import os
import sys
import json
import base64
import logging
import tempfile
import traceback
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Point, LineString
from skimage.morphology import remove_small_objects, disk
from skimage.segmentation import expand_labels
from skimage.filters.rank import modal
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import anthropic

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("GEOVECTOR_OS")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LegendUnit:
    code: str
    name: str

@dataclass
class LineFeature:
    feature_type: str
    geometry_wkt: str

@dataclass
class PointSymbol:
    symbol_type: str
    x: float
    y: float

@dataclass
class GEOVECTORConfig:
    input_image: str = "jangi_georef.tif"
    output_dir: str = "output_geovector"
    api_key: str = ""
    patch_size: int = 128
    epochs: int = 25
    batch_size: int = 16
    scratch_dir: str = tempfile.mkdtemp(prefix="geovector_")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 & 2 — PREPROCESSING & LLM INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
class LLMMapIntelligence:
    def __init__(self, cfg: GEOVECTORConfig):
        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=cfg.api_key) if cfg.api_key else None

    def _encode_image(self, img: np.ndarray) -> str:
        tmp = os.path.join(self.cfg.scratch_dir, "legend_crop.png")
        cv2.imwrite(tmp, img)
        with open(tmp, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def parse_legend(self, img_bgr: np.ndarray) -> list[LegendUnit]:
        if not self.client:
            log.warning("No Anthropic API key provided. Falling back to manual unit entry.")
            return self._manual_fallback()

        log.info("Sending map to Claude for automatic legend parsing...")
        # Crop bottom right quadrant assuming legend is there
        h, w = img_bgr.shape[:2]
        crop = img_bgr[int(h*0.5):, int(w*0.5):]
        b64 = self._encode_image(crop)

        prompt = """
        You are a geological map expert. Examine this geological map image.
        Find the legend, and extract the units.
        Respond ONLY with a JSON array: [{"code": "Qs", "name": "Quaternary Sands"}, ...]
        If you cannot clearly read a legend, output an empty array [].
        """
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            raw = response.content[0].text.strip().replace("```json", "").replace("```", "")
            data = json.loads(raw)
            if not data:
                log.warning("LLM could not confidently read legend. Falling back to manual.")
                return self._manual_fallback()

            units = [LegendUnit(code=d.get("code",""), name=d.get("name","")) for d in data]
            log.info(f"LLM extracted {len(units)} units automatically!")
            return units
        except Exception as e:
            log.warning(f"LLM Parsing failed ({e}). Falling back to manual entry.")
            return self._manual_fallback()

    def _manual_fallback(self) -> list[LegendUnit]:
        print("\n" + "="*40)
        print("  MANUAL UNIT CONFIGURATION")
        print("="*40)
        try:
            n = int(input("How many units are on this map? → "))
        except:
            n = 2
        units = []
        for i in range(n):
            name = input(f"Unit {i+1} Name/Code: ").strip() or f"Unit_{i+1}"
            units.append(LegendUnit(code=name[:4], name=name))
        return units


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — PYTORCH U-NET TEXTURE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class LiteUNet(nn.Module):
    def __init__(self, n_classes, in_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = self.conv1(torch.cat([x3, x], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x2, x], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x1, x], dim=1))
        return self.outc(x)

class GeoPatchDataset(Dataset):
    def __init__(self, image, rects_by_class, patch_size=128, samples_per_class=200):
        self.image = image
        self.is_rgb = len(image.shape) == 3
        self.patch_size = patch_size
        self.samples = []
        stride = patch_size // 4
        H, W = image.shape[:2]
        
        for c, rects in rects_by_class.items():
            class_samples = []
            for (rtop, rbot, cleft, cright) in rects:
                if rbot - rtop >= patch_size and cright - cleft >= patch_size:
                    for r in range(rtop, rbot - patch_size + 1, stride):
                        for c_col in range(cleft, cright - patch_size + 1, stride):
                            class_samples.append((r, c_col, c))
                else:
                    # Provide multiple jittered patches for small boxes to improve small unit learning
                    for _ in range(5):
                        center_r, center_c = (rtop + rbot) // 2, (cleft + cright) // 2
                        jitter_r = np.random.randint(-patch_size//4, patch_size//4 + 1)
                        jitter_c = np.random.randint(-patch_size//4, patch_size//4 + 1)
                        patch_r = max(0, center_r - patch_size // 2 + jitter_r)
                        patch_c = max(0, center_c - patch_size // 2 + jitter_c)
                        if patch_r + patch_size > H: patch_r = H - patch_size
                        if patch_c + patch_size > W: patch_c = W - patch_size
                        class_samples.append((c, rtop, rbot, cleft, cright, patch_r, patch_c))
            
            if len(class_samples) > 0:
                if len(class_samples) > samples_per_class:
                    # Subsample if too many
                    indices = np.random.choice(len(class_samples), samples_per_class, replace=False)
                    class_samples = [class_samples[i] for i in indices]
                elif len(class_samples) < samples_per_class:
                    # Upsample if too few
                    indices = np.random.choice(len(class_samples), samples_per_class, replace=True)
                    class_samples = [class_samples[i] for i in indices]
                self.samples.extend(class_samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        y = torch.full((self.patch_size, self.patch_size), 255, dtype=torch.long)
        if len(item) == 3:
            r, c_col, label = item
            if self.is_rgb:
                patch = self.image[r:r+self.patch_size, c_col:c_col+self.patch_size, :]
            else:
                patch = self.image[r:r+self.patch_size, c_col:c_col+self.patch_size]
            y.fill_(label)
        else:
            label, rect_r1, rect_r2, rect_c1, rect_c2, patch_r, patch_c = item
            if self.is_rgb:
                patch = self.image[patch_r:patch_r+self.patch_size, patch_c:patch_c+self.patch_size, :]
            else:
                patch = self.image[patch_r:patch_r+self.patch_size, patch_c:patch_c+self.patch_size]
            local_r1, local_r2 = rect_r1 - patch_r, rect_r2 - patch_r
            local_c1, local_c2 = rect_c1 - patch_c, rect_c2 - patch_c
            y[local_r1:local_r2, local_c1:local_c2] = label

        if np.random.rand() > 0.5:
            patch, y = np.flipud(patch), torch.flipud(y)
        if np.random.rand() > 0.5:
            patch, y = np.fliplr(patch), torch.fliplr(y)
            
        if self.is_rgb:
            x = torch.from_numpy(patch.copy()).float().permute(2, 0, 1)
        else:
            x = torch.from_numpy(patch.copy()).float().unsqueeze(0)
        return x, y

class UnetSegmenter:
    def __init__(self, cfg: GEOVECTORConfig):
        self.cfg = cfg

    def _collect_rectangles(self, display_img, title, H, W):
        all_rects = []
        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            r1, r2 = sorted([y1, y2])
            c1, c2 = sorted([x1, x2])
            r1, r2, c1, c2 = max(0, r1), min(H-1, r2), max(0, c1), min(W-1, c2)
            if (r2 - r1) > 10 and (c2 - c1) > 10:
                all_rects.append((r1, r2, c1, c2))
                print(f"       ✓ Collected! {r2-r1}x{c2-c1} px")
        fig, ax = plt.subplots(figsize=(14, 11))
        if len(display_img.shape) == 3:
            ax.imshow(display_img)
        else:
            ax.imshow(display_img, cmap='gray')
        ax.set_title(title + "\n(Draw ANY size box. Close window when done)", color='red')
        plt.tight_layout()
        selector = RectangleSelector(ax, on_select, useblit=True, button=[1], interactive=False)
        plt.show()
        return all_rects

    def execute(self, train_image: np.ndarray, display_image: np.ndarray, gray: np.ndarray, units: list[LegendUnit]) -> np.ndarray:
        H, W = gray.shape
        train_image_nn = (train_image.astype(np.float32) / 255.0)
        is_rgb = len(train_image_nn.shape) == 3
        in_channels = 3 if is_rgb else 1

        rects_by_class = {}
        class_counts = np.zeros(len(units))
        for i, unit in enumerate(units):
            rects = self._collect_rectangles(display_image, f"Draw TRAINING boxes inside '{unit.name}'", H, W)
            rects_by_class[i] = rects
            for r in rects:
                class_counts[i] += (r[1]-r[0]) * (r[3]-r[2])

        dataset = GeoPatchDataset(train_image_nn, rects_by_class, self.cfg.patch_size, samples_per_class=200)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        
        # Calculate Class Weights to prevent tiny units from being ignored!
        class_counts[class_counts == 0] = 1.0
        # More aggressive weighting (sqrt inverse) to handle massive class imbalance
        weights = 1.0 / (np.sqrt(class_counts) + 1e-5)
        weights = (weights / np.sum(weights)) * len(units)
        class_weights = torch.FloatTensor(weights).to(DEVICE)
        
        log.info(f"Training U-Net on {len(dataset)} augmented texture chips...")
        model = LiteUNet(len(units), in_channels=in_channels).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

        model.train()
        for ep in range(self.cfg.epochs):
            loss_m = 0.0
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                loss_m += loss.item()
            print(f"       Epoch {ep+1}/{self.cfg.epochs} — Loss: {(loss_m/len(loader)):.4f}", end="\r")
        print("\n")

        # Inference
        log.info("Running sliding-window neural inference...")
        model.eval()
        ps = self.cfg.patch_size
        stride = ps // 2
        pad_h, pad_w = (ps - H%ps)%ps, (ps - W%ps)%ps
        if is_rgb:
            padded = np.pad(train_image_nn, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            pH, pW = padded.shape[:2]
        else:
            padded = np.pad(train_image_nn, ((0, pad_h), (0, pad_w)), mode='reflect')
            pH, pW = padded.shape
        prob_map = np.zeros((len(units), pH, pW), dtype=np.float32)
        weight_map = np.zeros((pH, pW), dtype=np.float32)
        
        y_grid, x_grid = np.meshgrid(np.linspace(-1, 1, ps), np.linspace(-1, 1, ps))
        window_np = np.exp(-0.5 * (x_grid**2 + y_grid**2) / (0.5**2))
        win_t = torch.from_numpy(window_np).float().to(DEVICE)

        with torch.no_grad():
            for r in range(0, pH - ps + 1, stride):
                print(f"       Processing row {r}/{pH}...", end="\r")
                batches, c_coords = [], []
                for c in range(0, pW - ps + 1, stride):
                    if is_rgb:
                        batches.append(padded[r:r+ps, c:c+ps, :])
                    else:
                        batches.append(padded[r:r+ps, c:c+ps])
                    c_coords.append(c)
                if not batches: continue
                if is_rgb:
                    X = torch.from_numpy(np.stack(batches)).float().permute(0, 3, 1, 2).to(DEVICE)
                else:
                    X = torch.from_numpy(np.stack(batches)[:, np.newaxis, :, :]).float().to(DEVICE)
                probs = F.softmax(model(X), dim=1) * win_t
                for b_idx, c in enumerate(c_coords):
                    prob_map[:, r:r+ps, c:c+ps] += probs[b_idx].cpu().numpy()
                    weight_map[r:r+ps, c:c+ps] += window_np

        pred_map = np.argmax(prob_map / (weight_map + 1e-8), axis=0)[:H, :W]
        
        log.info("Applying topological nibble (filling gaps)...")
        bg_mask = gray > 250  # Increased threshold to prevent masking out light-colored units (like sand)
        pred_map[bg_mask] = -1
        clean_map = expand_labels((pred_map + 1).astype(np.int32), distance=150) - 1
        clean_map[bg_mask] = -1
        
        for uid in range(len(units)):
            mask = (clean_map == uid).astype(bool)
            mask_clean = remove_small_objects(mask, min_size=500) # Increased to 500 to merge messy edge polygons
            clean_map[(clean_map == uid) & (~mask_clean)] = -1
            
        clean_map = expand_labels((clean_map + 1).astype(np.int32), distance=200) - 1
        
        log.info("Smoothing vector topology boundaries via Modal rank filter...")
        # Rank filter applies majority vote smoothing while preserving perfect gapless boundaries
        shifted_map = (clean_map + 1).astype(np.uint8) 
        smooth_map = modal(shifted_map, disk(7)) 
        clean_map = smooth_map.astype(np.int32) - 1
        clean_map[bg_mask] = -1
        
        return clean_map


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — GEOPANDAS OPEN-SOURCE GIS INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
class GeoPandasIntegrator:
    def __init__(self, cfg: GEOVECTORConfig, transform, crs):
        self.cfg = cfg
        self.transform = transform
        self.crs = crs
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def vectorize_polygons(self, label_map: np.ndarray, units: list[LegendUnit]):
        log.info("Vectorizing Solid Geology Polygons...")
        vec_raster = (label_map + 1).astype(np.int32)
        features = []
        for geom, val in shapes(vec_raster, transform=self.transform):
            if val == 0: continue
            uid = int(val) - 1
            poly = shape(geom)
            if poly.is_valid and poly.area > 0:
                features.append({"geometry": poly, "UnitName": units[uid].name, "UnitCode": units[uid].code})
        
        if not features: return
        gdf = gpd.GeoDataFrame(features, geometry="geometry", crs=self.crs)
        gdf = gdf.dissolve(by="UnitName").reset_index()
        
        # Break MultiPolygons into separate discrete polygons
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        # Give each unique spatially continuous polygon its own ID
        gdf["FeatureID"] = gdf.index + 1
        
        # Apply topological polygon smoothing to eliminate raster saw-teeth artifacts
        import shapely
        pixel_size = max(abs(self.transform.a), abs(self.transform.e))
        tolerance = pixel_size * 5.0
        log.info(f"Applying topological coverage simplification (tolerance: {tolerance:.4f})")
        gdf["geometry"] = shapely.coverage_simplify(gdf.geometry.tolist(), tolerance)
        
        out_shp = os.path.join(self.cfg.output_dir, "GeologicalUnits.shp")
        gdf.to_file(out_shp)
        log.info(f"Saved {len(gdf)} dissolved polygons to {out_shp}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
class GEOVECTORPipelineOS:
    def __init__(self, cfg: GEOVECTORConfig):
        self.cfg = cfg

    def run(self):
        log.info("=" * 60)
        log.info("GEOVECTOR OS — Start")
        log.info("=" * 60)
        
        log.info("[1/5] Loading Raster...")
        with rasterio.open(self.cfg.input_image) as src:
            img = src.read()
            transform = src.transform
            crs = src.crs
            try:
                cmap = src.colormap(1)
            except ValueError:
                cmap = None
            
        if img.shape[0] >= 3:
            img_bgr = cv2.cvtColor(img[:3].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            ans = input("This map appears to be colored. Train the AI on full color RGB instead of Grayscale? (y/n): ").strip().lower()
            if ans == 'y':
                train_image = img_bgr 
                display_image = cv2.medianBlur(img_rgb, 5) # Draw on colored map!
            else:
                train_image = gray
                display_image = cv2.medianBlur(gray, 5)
        elif cmap:
            log.info("Indexed Colormap detected! Unpacking 1-band palette to full RGB...")
            rgb = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
            for k, v in cmap.items():
                rgb[img[0] == k] = v[:3]
            img_rgb = rgb
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            ans = input("This 1-band map contains an embedded Colormap. Train the AI on full color RGB instead of Grayscale? (y/n): ").strip().lower()
            if ans == 'y':
                train_image = img_bgr 
                display_image = cv2.medianBlur(img_rgb, 5) 
            else:
                train_image = gray
                display_image = cv2.medianBlur(gray, 5)
        else:
            img_bgr = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)
            gray = img[0]
            train_image = gray
            display_image = cv2.medianBlur(gray, 5)

        log.info("[2/5] Map Intelligence & Legend Parsing")
        llm = LLMMapIntelligence(self.cfg)
        units = llm.parse_legend(img_bgr)
        
        log.info("[3/5] Deep Learning Texture Segmentation")
        segmenter = UnetSegmenter(self.cfg)
        label_map = segmenter.execute(train_image, display_image, gray, units)
        
        log.info("[4/5] Evaluating and vectorizing solid shapes...")
        # Inject Accuracy Assessment here!
        self._assess_accuracy(train_image, display_image, gray, label_map, units)
        
        log.info("[5/5] Open-Source GIS Integration")
        gis = GeoPandasIntegrator(self.cfg, transform, crs)
        gis.vectorize_polygons(label_map, units)
        
        log.info("=" * 60)
        log.info("GEOVECTOR OS COMPLETE. Check 'output_geovector' folder.")
        log.info("=" * 60)

    def _assess_accuracy(self, train_image, display_image, gray, clean_map, units):
        print("\n" + "="*40)
        ans = input("Do you want to run Accuracy Assessment on the map? (y/n): ").strip().lower()
        if ans != 'y': return
        
        log.info("Draw TEST rectangles. Do NOT draw on regions you used in training!")
        segmenter = UnetSegmenter(self.cfg)
        y_true, y_pred = [], []
        H, W = gray.shape
        
        for i, unit in enumerate(units):
            rects = segmenter._collect_rectangles(display_image, f"TEST Boxes for '{unit.name}'", H, W)
            for (r1, r2, c1, c2) in rects:
                mask = clean_map[r1:r2, c1:c2]
                valid = mask != -1
                pred_vals = mask[valid]
                y_true.extend([i] * len(pred_vals))
                y_pred.extend(pred_vals)
                
        if len(y_true) > 0:
            print("\n---------- ACCURACY REPORT ----------")
            try:
                from sklearn.metrics import classification_report, confusion_matrix
                # Only use target names that actually appeared in the ground truth
                present_labels = sorted(list(set(y_true)))
                names = [units[idx].name for idx in present_labels]
                
                print(classification_report(y_true, y_pred, labels=present_labels, target_names=names, zero_division=0))
                cm = confusion_matrix(y_true, y_pred, labels=present_labels)
                print("Confusion Matrix:\n", cm)
            except Exception as e:
                print(f"Metrics print failed (sklearn not installed or label mismap): {e}")
                print(f"Overall Accuracy: {np.mean(np.array(y_true) == np.array(y_pred))*100:.1f}%")
        else:
            print("No test data collected.")


if __name__ == "__main__":
    print("\n--- GEOVECTOR OS Initialization ---")
    api_key = input("Enter Anthropic API Key for Legend Parsing (Press Enter to skip & manual type): ").strip()
    
    cfg = GEOVECTORConfig(
        input_image="jangi_georef.tif",
        output_dir="output_geovector",
        api_key=api_key,
        patch_size=128,
        epochs=15,    # Slightly lower for speed
        batch_size=16
    )
    
    pipeline = GEOVECTORPipelineOS(cfg)
    pipeline.run()
