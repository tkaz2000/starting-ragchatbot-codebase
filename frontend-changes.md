# Frontend Changes: Prettier Code Formatting

## Summary

Added Prettier as the code formatter for the frontend (`frontend/index.html`, `frontend/script.js`, `frontend/style.css`) and applied it once so the whole frontend follows one style. The backend (Python) is not touched. Black was not used because it only formats Python.

## What changed

- **`package.json`** (new): Node tooling manifest. Prettier is pinned to the exact version `3.3.3` in `devDependencies`. It has no runtime dependencies.
  - `npm run format`: rewrites `frontend/**/*.{html,css,js}` in place.
  - `npm run format:check`: fails if any frontend file is not formatted. Use this for CI or before committing.
  - `npm run check`: alias for `format:check`.
- **`package-lock.json`** (new): lockfile for the pinned Prettier install.
- **`.prettierrc.json`** (new): formatting rules matched to the existing code so the diff stays small:
  - 4-space indentation, single quotes, semicolons, `printWidth` 100, `trailingComma` `es5`, LF line endings.
- **`.prettierignore`** (new): limits Prettier to the frontend. It excludes `backend/`, `docs/`, `.claude/`, `.trees/`, `.playwright-mcp/`, `uv.lock`, and `node_modules/`.
- **`.gitignore`**: added `node_modules/`.
- **`frontend/index.html`**: reformatted. Changes are whitespace and attribute layout only. Long tags are wrapped one attribute per line, void elements are self-closed (`<input ... />`), and `<!DOCTYPE html>` is lowercased to `<!doctype html>`. Quote style changed on the `data-question` attributes that contain double quotes (they now use single quotes). Rendered output is unchanged.
- **`frontend/script.js`**: reformatted (spacing, line wrapping, trailing commas). No logic changes. `node --check` passes.
- **`frontend/style.css`**: reformatted (spacing and rule layout). No selector or value changes.

## Verification

- `npm run check` reports "All matched files use Prettier code style!"
- `node --check frontend/script.js` passes.
- A whitespace-insensitive diff (`git diff -w`) confirms the changes are formatting-only apart from the `index.html` attribute and quote changes listed above.

## How to use

```bash
npm install            # first time only, installs the pinned Prettier
npm run format         # format the frontend
npm run check          # verify formatting without changing files
```

## Notes

- Prettier formats only the frontend. Backend Python formatting (Black) is a separate decision and is not part of this change.
- The app still has no automated tests. `npm run check` only verifies formatting.
