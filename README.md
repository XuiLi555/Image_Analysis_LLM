# Image_Analysis_LLM


Analyze images with a multimodal large language model. Give ImageLens a photo
and it can **describe** it, **answer questions** about it, **read text** in it,
**classify** it, **detect** objects, or **compare** two images — all through a
clean Python API and a command-line tool.

The default backend is the **Anthropic Claude** vision API, but the provider
layer is pluggable. A built-in **offline mock provider** means the tests, demo,
and CLI all run with **no API key**, so you can explore the project before
adding credentials.

---

## Features

| Task | Method | CLI |
|---|---|---|
| Describe / caption | `describe(image, detail="brief"\|"detailed")` | `imagelens describe img.jpg` |
| Visual Q&A (VQA) | `ask(image, question)` | `imagelens ask img.jpg "..."` |
| Read text (OCR-style) | `extract_text(image)` | `imagelens text img.jpg` |
| Classify into labels | `classify(image, labels)` → JSON | `imagelens classify img.jpg --labels a b c` |
| Detect / count objects | `detect(image, objects)` | `imagelens detect img.jpg --objects "..."` |
| Compare two images | `compare(image_a, image_b)` | `imagelens compare a.jpg b.jpg` |

Inputs can be a **file path**, an **`http(s)` URL**, raw **bytes**, or an
`Image` object. Over-large images are automatically resized before upload.
