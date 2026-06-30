# Whiteboard 3

Whiteboard 3 is a Vite + React frontend for a large pannable math whiteboard. React owns UI state and composition; plain JavaScript modules own canvas drawing, stroke smoothing, erasing, history, viewport math, and problem-flow geometry.

## Run Locally

Install dependencies once:

```sh
npm install
```

Start the dev server:

```sh
npm run dev
```

The app is served by Vite. Open the URL printed in the terminal, usually `http://localhost:5500/`. If another server already uses port `5500`, Vite may choose another port.

To start the OCR server, Whiteboard API, and app together while OCR still lives
in `../whiteboard_2`, run:

```sh
npm run dev:all
```

Leave that terminal open. Press `Ctrl+C` there to stop all three processes.

If something is already using the local dev ports, stop the old servers with:

```sh
npm run dev:stop
```

## Recognition API

OCR, semantic scoring, and grading requests go through a FastAPI backend. During
local experiments, run the CoMER/DBNet model API on port `8000`, then start the
whiteboard API on port `8010`:

```sh
python3 -m src.server.app \
  --port 8010 \
  --upstream-api-url http://127.0.0.1:8000 \
  --semantic-timeout 2.5
```

The browser defaults to the current page hostname on port `8010`. Set
`VITE_API_URL` to override it:

```sh
VITE_API_URL=http://127.0.0.1:8010 npm run dev
```

## Chromebook Access On The Same Wi-Fi

The dev script binds Vite to `0.0.0.0`, so another device on the same network can open it.

1. Start the dev server with `npm run dev`.
2. Find the Mac's Wi-Fi IP address:

```sh
ipconfig getifaddr en0
```

3. On the Chromebook, open `http://<mac-ip>:<vite-port>/`.

Example: if the Mac IP is `192.168.1.156` and Vite is running on `5500`, open `http://192.168.1.156:5500/`.

If the page does not load, confirm the Chromebook is on the same Wi-Fi network and use the exact port printed by Vite.

## Keyboard Shortcuts

- `P`: pen tool
- `M`: mouse/pan tool
- `E`: eraser tool
- `Ctrl+Z` / `Cmd+Z`: undo
- `Ctrl+Shift+Z` / `Cmd+Shift+Z` or `Ctrl+Y` / `Cmd+Y`: redo
- `Shift+C`: clear strokes
- `Ctrl+B` / `Cmd+B`: toggle developer debug boxes

## Debug Boxes

`Ctrl+B` / `Cmd+B` toggles a developer-only overlay. Blue dashed boxes show the invisible problem catchment region. Green boxes show the active dynamic answer region. Purple boxes show submitted/frozen answer regions.

The overlay is non-interactive and pans with the whiteboard.
