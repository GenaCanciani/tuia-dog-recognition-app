from __future__ import annotations

import json
import os
from typing import Any

import cv2
import gradio as gr
import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_CHOICES = [
    m.strip()
    for m in os.environ.get("MODEL_CHOICES", "baseline").split(",")
    if m.strip()
]
CLASSIFIER_CHOICES = [m for m in MODEL_CHOICES if m != "baseline"] or [
    "resnet18_finetuned",
    "cnn_custom",
]


def _client() -> httpx.Client:
    return httpx.Client(timeout=120.0)


def _abs_url(rel: str | None) -> str | None:
    if not rel:
        return None
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    return f"{API_BASE}{rel}" if rel.startswith("/") else f"{API_BASE}/{rel}"


def upload_numpy_image(image: np.ndarray | None) -> str:
    if image is None:
        raise ValueError("Falta la imagen.")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen.")
    data = buf.tobytes()
    files = {"file": ("upload.jpg", data, "image/jpeg")}
    with _client() as c:
        r = c.post(f"{API_BASE}/upload", files=files)
        r.raise_for_status()
        body = r.json()
    return str(body["path"])


def draw_detections_on_bgr(image_bgr: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Dibuja bounding boxes, raza predicha y scores; retorna RGB para gradio."""
    vis = image_bgr.copy()
    for det in result.get("detections", []):
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        breed = det.get("breed", "?")
        det_score = det.get("det_score", 0.0)
        breed_score = det.get("breed_score", 0.0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 220, 80), 2)
        txt = f"{breed} det:{det_score} cls:{breed_score}"
        cv2.putText(
            vis,
            txt,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 220, 80),
            2,
        )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def decode_image_bytes(content: bytes) -> np.ndarray | None:
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _download_image(url: str | None) -> np.ndarray | None:
    full = _abs_url(url)
    if not full:
        return None
    try:
        with _client() as c:
            r = c.get(full)
            r.raise_for_status()
        return decode_image_bytes(r.content)
    except httpx.HTTPError:
        return None


def start_search(image: np.ndarray | None, model: str, top_k: float) -> tuple[str, str]:
    try:
        path = upload_numpy_image(image)
        with _client() as c:
            r = c.post(
                f"{API_BASE}/search",
                json={"source_path": path, "model": model, "top_k": int(top_k)},
            )
            r.raise_for_status()
            job_id = r.json()["job_id"]
        msg = (
            f"Busqueda encolada. **job_id:** `{job_id}`\n\n"
            f"Modelo de embeddings: `{model}` — Backend: `{API_BASE}`\n\n"
            "Pulsa **Consultar resultado de este job** o ve a **Estado y resultados**."
        )
        return job_id, msg
    except httpx.HTTPStatusError as exc:
        return "", f"Error HTTP: {exc.response.status_code} — {exc.response.text[:500]}"
    except Exception as exc:
        return "", f"Error: {exc}"


def start_classify(image: np.ndarray | None, model: str) -> tuple[str, str]:
    try:
        path = upload_numpy_image(image)
        with _client() as c:
            r = c.post(
                f"{API_BASE}/classify",
                json={"source_path": path, "model": model},
            )
            r.raise_for_status()
            job_id = r.json()["job_id"]
        msg = (
            f"Clasificacion encolada. **job_id:** `{job_id}`\n\n"
            f"Modelo entrenado: `{model}`\n\n"
            "Pulsa **Consultar resultado de este job** o ve a **Estado y resultados**."
        )
        return job_id, msg
    except httpx.HTTPStatusError as exc:
        return "", f"Error HTTP: {exc.response.status_code} — {exc.response.text[:500]}"
    except Exception as exc:
        return "", f"Error: {exc}"


def start_detect(image: np.ndarray | None) -> tuple[str, str]:
    try:
        path = upload_numpy_image(image)
        with _client() as c:
            r = c.post(f"{API_BASE}/detect", json={"source_path": path})
            r.raise_for_status()
            job_id = r.json()["job_id"]
        msg = (
            f"Deteccion encolada. **job_id:** `{job_id}`\n\n"
            "Pulsa **Consultar resultado de este job** o ve a **Estado y resultados**."
        )
        return job_id, msg
    except httpx.HTTPStatusError as exc:
        return "", f"Error HTTP: {exc.response.status_code} — {exc.response.text[:500]}"
    except Exception as exc:
        return "", f"Error: {exc}"


def _render_search(data: dict[str, Any], source_image_url: str | None, links_md: str):
    query_bgr = _download_image(source_image_url)
    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB) if query_bgr is not None else None

    gallery: list[tuple[np.ndarray, str]] = []
    for n in data.get("neighbors", []):
        img_bgr = _download_image(n.get("url"))
        if img_bgr is None:
            continue
        caption = f"{n.get('breed', '?')} ({n.get('score', 0.0)})"
        gallery.append((cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption))

    breed = data.get("predicted_breed", "?")
    score = data.get("score", 0.0)
    model = data.get("model", "?")
    extra = (
        f"**Raza predicha:** {breed} (score: {score})\n\n"
        f"**Modelo de embeddings:** `{model}`\n\n{links_md}"
    )
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    return query_rgb, gallery, pretty, extra, "**Estado:** completado (busqueda por similitud)."


def _render_classify(data: dict[str, Any], source_image_url: str | None, links_md: str):
    img_bgr = _download_image(source_image_url)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
    breed = data.get("breed", "?")
    score = data.get("score", 0.0)
    model = data.get("model", "?")
    extra = (
        f"**Raza predicha:** {breed} (score: {score})\n\n"
        f"**Modelo entrenado:** `{model}`\n\n{links_md}"
    )
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    return img_rgb, [], pretty, extra, "**Estado:** completado (clasificacion supervisada)."


def _render_detect(data: dict[str, Any], source_image_url: str | None, links_md: str):
    img_bgr = _download_image(source_image_url)
    if img_bgr is None:
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        return None, [], pretty, links_md, "**Estado:** completado; no se decodifico la imagen origen."
    vis = draw_detections_on_bgr(img_bgr, data)
    breeds = ", ".join(data.get("detected_breeds") or []) or "(ninguna)"
    extra = f"**Razas detectadas:** {breeds}\n\n{links_md}"
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    return vis, [], pretty, extra, "**Estado:** completado (deteccion y clasificacion)."


def consult_status(job_id: str) -> tuple[np.ndarray | None, list, str, str, str]:
    raw = (job_id or "").strip()
    if not raw:
        return None, [], "", "", "Ingresa un job_id."

    try:
        with _client() as c:
            r = c.get(f"{API_BASE}/status/{raw}")
            if r.status_code == 404:
                return None, [], "", "", "job_id no encontrado."
            r.raise_for_status()
            st = r.json()
    except httpx.HTTPError as exc:
        return None, [], "", "", f"No se pudo consultar el estado: {exc}"

    status = st.get("status")
    if status == "inProgress":
        return None, [], "", "", "**Estado:** en progreso…"

    if status == "failed":
        reason = st.get("reason") or "sin detalle"
        return None, [], "", "", f"**Estado:** fallido\n\n`{reason}`"

    artifact_url = st.get("artifact_url")
    source_image_url = st.get("source_image_url")
    link = st.get("link") or ""

    links_lines = []
    au = _abs_url(artifact_url)
    su = _abs_url(source_image_url)
    if au:
        links_lines.append(f"- [Descargar resultado JSON]({au})")
    if su:
        links_lines.append(f"- [Descargar imagen origen]({su})")
    links_md = "\n".join(links_lines) if links_lines else ""

    if not artifact_url and link not in ("", "none"):
        return None, [], "", links_md, f"**Estado:** hecho, sin URL publica para: `{link}`"

    if not artifact_url:
        return None, [], "", links_md, "**Estado:** hecho, pero sin enlace de descarga."

    try:
        with _client() as c:
            ar = c.get(_abs_url(artifact_url) or "")
            ar.raise_for_status()
            content = ar.content
    except httpx.HTTPError as exc:
        return None, [], "", links_md, f"Error al bajar el resultado: {exc}"

    try:
        data = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, [], "", links_md, "No se pudo parsear el JSON de resultado."

    result_type = data.get("type")
    if result_type == "search":
        return _render_search(data, source_image_url, links_md)
    if result_type == "classify":
        return _render_classify(data, source_image_url, links_md)
    if result_type == "detect":
        return _render_detect(data, source_image_url, links_md)

    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    return None, [], pretty, links_md, "**Estado:** completado (resultado sin tipo conocido)."


def build_ui() -> gr.Blocks:
    title = os.environ.get("APP_NAME", "Reconocimiento de razas de perros") + " — UI (externa)"
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "### Deteccion y clasificacion de razas de perros (cliente HTTP)\n"
            f"Backend configurado: `{API_BASE}` (`BACKEND_URL`). "
            "Sube imagenes, obtén **job_id** y consulta el estado. "
            "Las descargas usan los endpoints `/files/output/...` y `/files/data/...`. "
            "**Etapa 1:** imagen consultada, top K similares y raza predicha. "
            "**Etapa 2:** clasificacion supervisada con el modelo entrenado. "
            "**Etapa 3:** bounding boxes, raza y scores de confianza."
        )

        job_id_shared = gr.Textbox(label="Job ID (ultimo o manual)", lines=1)

        with gr.Tab("Busqueda por similitud (Etapa 1)"):
            gr.Markdown("Llama a `POST /upload` y `POST /search`.")
            search_in = gr.Image(label="Imagen de un perro", type="numpy", height=320)
            search_model = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES[0] if MODEL_CHOICES else None,
                label="Modelo de embeddings",
            )
            search_topk = gr.Slider(1, 20, value=10, step=1, label="Top K")
            search_btn = gr.Button("Buscar similares", variant="primary")
            search_log = gr.Markdown()
            search_quick = gr.Button("Consultar resultado de este job")

        with gr.Tab("Clasificacion supervisada (Etapa 2)"):
            gr.Markdown(
                "Llama a `POST /upload` y `POST /classify`. Clasifica la imagen completa "
                "con el modelo entrenado en la Etapa 2 (reutiliza `classify_detected_dog`, "
                "por lo que requiere tambien esa funcion de la Etapa 3)."
            )
            cls_in = gr.Image(label="Imagen de un perro", type="numpy", height=320)
            cls_model = gr.Dropdown(
                choices=CLASSIFIER_CHOICES,
                value=CLASSIFIER_CHOICES[0],
                label="Modelo entrenado",
            )
            cls_btn = gr.Button("Clasificar", variant="primary")
            cls_log = gr.Markdown()
            cls_quick = gr.Button("Consultar resultado de este job")

        with gr.Tab("Deteccion y clasificacion (Etapa 3)"):
            gr.Markdown("Llama a `POST /upload` y `POST /detect`.")
            det_in = gr.Image(label="Imagen (uno o varios perros)", type="numpy", height=320)
            det_btn = gr.Button("Detectar y clasificar", variant="primary")
            det_log = gr.Markdown()
            det_quick = gr.Button("Consultar resultado de este job")

        with gr.Tab("Estado y resultados"):
            gr.Markdown("`GET /status/{job_id}` — enlaces de descarga en el texto si aplica.")
            status_in = gr.Textbox(label="job_id a consultar", lines=1)
            status_btn = gr.Button("Consultar", variant="primary")
            status_line = gr.Markdown()
            vis_out = gr.Image(label="Imagen (consulta o detecciones)", height=420)
            gallery_out = gr.Gallery(label="Top K imagenes similares", columns=5, height=360)
            json_out = gr.Code(label="JSON (resultado)", language="json", lines=16)
            extra_md = gr.Markdown()

        def _on_search(
            img: np.ndarray | None, model: str, top_k: float
        ) -> tuple[str, str, str, None, list, str]:
            jid, msg = start_search(img, model, top_k)
            return jid, msg, jid, None, [], ""

        def _on_classify(
            img: np.ndarray | None, model: str
        ) -> tuple[str, str, str, None, list, str]:
            jid, msg = start_classify(img, model)
            return jid, msg, jid, None, [], ""

        def _on_detect(img: np.ndarray | None) -> tuple[str, str, str, None, list, str]:
            jid, msg = start_detect(img)
            return jid, msg, jid, None, [], ""

        search_btn.click(
            _on_search,
            [search_in, search_model, search_topk],
            [job_id_shared, search_log, status_in, vis_out, gallery_out, json_out],
        )
        cls_btn.click(
            _on_classify,
            [cls_in, cls_model],
            [job_id_shared, cls_log, status_in, vis_out, gallery_out, json_out],
        )
        det_btn.click(
            _on_detect,
            det_in,
            [job_id_shared, det_log, status_in, vis_out, gallery_out, json_out],
        )

        def _consult(from_id: str) -> tuple[Any, list, str, str, str]:
            vis, gallery, js, extra, line = consult_status(from_id)
            return vis, gallery, js, extra, line

        status_btn.click(
            _consult,
            status_in,
            [vis_out, gallery_out, json_out, extra_md, status_line],
        )
        search_quick.click(
            _consult,
            job_id_shared,
            [vis_out, gallery_out, json_out, extra_md, status_line],
        )
        cls_quick.click(
            _consult,
            job_id_shared,
            [vis_out, gallery_out, json_out, extra_md, status_line],
        )
        det_quick.click(
            _consult,
            job_id_shared,
            [vis_out, gallery_out, json_out, extra_md, status_line],
        )

    return demo
