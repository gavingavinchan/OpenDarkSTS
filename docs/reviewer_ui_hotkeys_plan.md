# Plan: Fix reviewer UI — hotkeys, tooltips, autosave

## Context

The visual reviewer works but is clunky to use. The user needs keyboard-driven workflow for efficient review of 700+ tiles, tooltip discoverability for all buttons, and automatic geometry saving so edits aren't lost.

## Files to modify

- `scripts/review_visual_detections_ui.py` — all changes (HTML/CSS/JS in the embedded template)
- `scripts/visual_review_lib.py` — `delete_detection()` function

## 1. Hotkey Mapping

Left hand on keyboard, right hand on mouse. Single-key, mnemonic shortcuts.

| Key | Action | When |
|-----|--------|------|
| `Space` | Advance review cycle | Always |
| `Left` | Previous detection / retreat review | Always |
| `Right` | Next detection / advance review | Always |
| `Tab` | Cycle label: Ship → Ships → Neither | Detection selected |
| `S` | Label Ships (STS) | Detection selected |
| `D` | Label Ship (non-STS) | Detection selected |
| `N` | Mark Neither (background/false-positive) | Detection selected |
| `Del`/`Backspace` | Delete box permanently | Detection selected |
| `[` | Previous image | Always |
| `]` | Next image | Always |
| `B` | Draw new Box | Always |
| `E` | Toggle Edit on selected | Detection selected |
| `G` | Save Geometry | Detection selected & editing |
| `Escape` | Cancel draw or clear selection | Always |
| `C` | Complete Image | Always |
| `?` | Toggle hotkey cheat-sheet overlay | Always |

## 2. Review Cycle (Space bar flow)

For each image, the review follows this cycle:

1. **Zoomed-out overview** — user sees the full tile to get context
2. Press **Space** → zooms into **first bounding box**, selects it
3. User labels with **Tab** (cycle) or **S**/**D**/**N** keys
4. Press **Space** → zooms into **next bounding box**
5. Repeat until all boxes reviewed
6. Press **Space** → **zoomed-out overview** again for final check
7. Press **Space** → moves to **next image**, starts at overview

**Left/Right arrows** also navigate forward/backward through this cycle.
Visual arrow buttons are shown at the bottom-right of the map.

## 3. Tab — Cycle Label on Selected Box

When a box is selected, pressing Tab cycles through Ship → Ships → Neither without deselecting or navigating away. The box stays selected and in view so the user can see the color/label change.

When no box is selected, Tab toggles the class dropdown for new boxes.

## 4. Tooltip System

CSS-only tooltips using `data-tooltip` attribute + `::after` pseudo-element.

- Every button gets `data-tooltip="Description [Key]"`
- Appears instantly on hover, disappears on mouse leave
- Styled: dark bg, white text, rounded, positioned above the button

## 5. Hotkey Cheat-Sheet Overlay

- Hidden `<div id="hotkeyOverlay">` in the body, toggled by `?`
- Fixed position, centered, semi-transparent backdrop
- Two-column grid of all shortcuts
- Click backdrop or press `?` again to dismiss

## 6. Detection Review Progress Bar

- Add a bottom-center progress bar overlay to the map
- The bar is discrete: one visible segment per active detection in the current image
- Fill one segment per manually reviewed or traversed detection in the active review cycle
- Show a count label such as `5/10` on the right side of the bar
- During the `Space` review cycle, each advance to the next detection increments the displayed progress by one until the image reaches final overview
- Keep the bar compact and flat, without an extra title label above it
- Reconstruct the displayed progress from persisted review state after reload so the bar resumes instead of resetting

## 7. Scroll Behavior

- The page itself should stay fixed vertically during review
- Mouse wheel over the image viewer should zoom the Leaflet image only
- Mouse wheel over the sidebar should scroll only the sidebar content

## 8. Geometry Autosave

`autoSaveGeometryIfNeeded()` saves detection geometry automatically before:

- `selectLayer()` — switching to a different detection
- `clearSelection()` — deselecting
- `openImage()` — navigating to a different image
- `navigateDetail()` — stepping through detections

## 9. Delete Box vs Mark Neither

- **Mark Neither** (`N`) — keeps the detection as `is_active=0, review_label='neither'` for false-positive training data
- **Delete Box** (`Del`/`Backspace`) — permanently removes the detection row from the DB

Backend: `POST /api/delete-detection`, `delete_detection()` in `visual_review_lib.py`.

## Verification

1. Reload the reviewer, press `?` — cheat sheet appears with all shortcuts
2. Hover any button — tooltip with description and shortcut appears
3. Click a detection, drag a corner to resize, press `]` to go to next image — geometry auto-saves
4. Press Space repeatedly — cycles: overview → box1 → box2 → ... → overview → next image
5. Left/Right arrows navigate the same cycle
6. The bottom-center discrete progress bar advances one segment at a time during the review cycle and shows the matching `reviewed/total` count on the right side
7. Mouse wheel over the image zooms the image only, while mouse wheel over the sidebar scrolls only the sidebar
8. Select a box, press Tab — label cycles Ship → Ships → Neither, box stays selected
9. Use `S`/`D`/`N` keys to label detections directly
10. Use `[`/`]` to jump between images
11. Press `Del` — box is permanently deleted from DB
12. Press `Escape` to cancel drawing or clear selection
