# Wireframes

The drawings each interface was built from. Every `.excalidraw` here is the
editable source; the `.png` beside it is a render for reading in a diff or a
browser.

| File | What it describes |
|---|---|
| `pages_report_wireframe` | the GitHub Pages dashboard, as first laid out |
| `streamlit_app_wireframe` | the Streamlit app and its six tabs |
| `map_feature_wireframe` | the "Where the data comes from" globe and city list |
| `method_section_wireframe` | the "How this works" section — loop, model, parameters |

These live in `docs/` rather than `site/` on purpose: everything under `site/`
is uploaded as the published Pages artifact, and the drawings are working
material, not part of the site.

## Working with them

They are drawn and read back through
[mcp-excalidraw-server](https://github.com/ldele/mcp_excalidraw), which reads a
canvas *as an interface* — screens, nesting, component roles, reading order,
navigation — rather than as a list of shapes:

```bash
npx -y mcp-excalidraw-server import docs/wireframes/map_feature_wireframe.excalidraw --replace
npx -y mcp-excalidraw-server wireframe    # read it back as a UI
```

The drawing conventions they follow (layout grid, the palette split that makes
role inference work, when to declare a `role`) are documented in that project
under `skills/excalidraw-skill/references/wireframe-conventions.md`. The short
version: a wireframe is finished when `wireframe` reads it back as what you
meant, not when the screenshot looks right.
