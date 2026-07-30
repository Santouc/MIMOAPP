import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
logging.disable(logging.CRITICAL)

import numpy as np
from PySide6.QtWidgets import QApplication

from app.app_context import AppContext
from app.screens.translate_screen import TranslateScreen
from core.preprocessing import normalize_landmarks
from ml.dynamic_classifier import DynamicSignClassifier
from services.path_service import PathService
from services.training_service import TrainingService

RNG = np.random.default_rng(42)
FPS = 30.0


def pose(dataset, letra, classifier):
    labels, y = dataset["labels"], dataset["y"]
    for i in range(len(dataset["X"])):
        if labels[y[i]] == letra:
            candidate = np.array(dataset["X"][i], dtype=np.float32)
            sign, confidence = classifier.classify([candidate.tolist()])
            if sign == letra and confidence > 0.9:
                return candidate
    raise ValueError(f"sin muestra confiable para {letra}")


def still(p, frames, sigma=0.0015):
    return [p + RNG.normal(0, sigma, p.shape).astype(np.float32) for _ in range(frames)]


def transition(a, b, frames=12):
    return [a + (b - a) * (t / (frames - 1)) for t in range(frames)]


def run(screen, ctx, frames, label):
    ctx.transcription.reset()
    screen.sign_buffer.clear()
    screen.dynamic_sequence.clear()
    screen.dynamic_buffer.clear()
    screen.still_frames = 0
    screen._last_motion_reference = None
    screen._last_seen_accepted_at = 0.0
    t = 1000.0
    state = None
    for frame in frames:
        landmarks = [frame.tolist()]
        screen._update_motion_state(landmarks)
        static_sign = screen._process_static_prediction(landmarks)
        dynamic_sign = screen._process_dynamic_prediction(landmarks)
        final_sign = screen._select_final_sign(static_sign, dynamic_sign)
        state = ctx.transcription.process_sign(final_sign, t)
        accepted_at = ctx.transcription.last_accepted_at
        if accepted_at != screen._last_seen_accepted_at:
            screen._last_seen_accepted_at = accepted_at
            screen._reset_dynamic_state()
        t += 1.0 / FPS
    print(f"{label:52s} letras registradas: '{state.raw_text}'")
    return state.raw_text


def smoothstep(values):
    return values * values * (3.0 - 2.0 * values)


def circle_seq(base, rng, frames=20):
    radius = rng.uniform(0.09, 0.16)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    angles = phase + np.linspace(0.0, 2.0 * np.pi, frames)
    dx = np.cos(angles) * radius
    dy = np.sin(angles) * radius
    dx -= dx[0]
    dy -= dy[0]
    seq = np.repeat(base[None, :, :], frames, axis=0).copy()
    seq[:, :, 0] += dx[:, None]
    seq[:, :, 1] += dy[:, None]
    return (seq + rng.normal(0, 0.002, seq.shape)).astype(np.float32)


def line_seq(base, rng, frames=20):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    length = rng.uniform(0.2, 0.35)
    progress = smoothstep(np.linspace(0.0, 1.0, frames))
    seq = np.repeat(base[None, :, :], frames, axis=0).copy()
    seq[:, :, 0] += (np.cos(angle) * length * progress)[:, None]
    seq[:, :, 1] += (np.sin(angle) * length * progress)[:, None]
    return (seq + rng.normal(0, 0.002, seq.shape)).astype(np.float32)


def transition_seq(a, b, rng, frames=20):
    t = smoothstep(np.linspace(0.0, 1.0, frames))[:, None, None]
    seq = a[None, :, :] * (1 - t) + b[None, :, :] * t
    return (seq + rng.normal(0, 0.002, seq.shape)).astype(np.float32)


def drift_seq(base, rng, frames=20):
    steps = rng.normal(0.0, 1.0, size=(frames, 2))
    path = np.cumsum(steps, axis=0)
    path -= path[0]
    span = max(float(np.max(np.linalg.norm(path, axis=1))), 1e-6)
    path *= rng.uniform(0.1, 0.3) / span
    seq = np.repeat(base[None, :, :], frames, axis=0).copy()
    seq[:, :, 0] += path[:, 0][:, None]
    seq[:, :, 1] += path[:, 1][:, None]
    return (seq + rng.normal(0, 0.002, seq.shape)).astype(np.float32)


def shapes_by_letter(static_ds, letra):
    labels, y = static_ds["labels"], static_ds["y"]
    return [np.array(x, dtype=np.float32) for x, yy in zip(static_ds["X"], y) if labels[yy] == letra]


def build_trajectory_workspace(static_ds, rng):
    temp_dir = Path(tempfile.mkdtemp(prefix="tls_trayectoria_"))
    datasets_dir = temp_dir / "data" / "datasets"
    datasets_dir.mkdir(parents=True)
    shutil.copy("data/datasets/dataset_static.json", datasets_dir / "dataset_static.json")
    shapes = shapes_by_letter(static_ds, "L")
    X, y = [], []
    for _ in range(12):
        X.append(circle_seq(shapes[rng.integers(len(shapes))], rng).tolist())
        y.append(0)
        X.append(line_seq(shapes[rng.integers(len(shapes))], rng).tolist())
        y.append(1)
    dataset = {"X": X, "y": y, "labels": ["CIRCULO", "LINEA"], "metadata": {"type": "dynamic"}}
    (datasets_dir / "dataset_dynamic.json").write_text(json.dumps(dataset), encoding="utf-8")
    return temp_dir


def eval_hits(clf, sequences, expected):
    return sum(1 for seq in sequences if clf.classify_sequence(list(seq))[0] == expected)


def train_baseline_vieja_normalizacion(temp_dir):
    ds = json.load(open(temp_dir / "data" / "datasets" / "dataset_dynamic.json", encoding="utf-8"))
    X = np.array(ds["X"], dtype=np.float32)
    y = np.array(ds["y"], dtype=np.int32)
    n, t, p, c = X.shape
    Xn = normalize_landmarks(X.reshape(n * t, p, c)).reshape(n, t, p, c)
    svc = TrainingService(PathService(temp_dir))
    model = svc._create_dynamic_model(t, 2)
    order = np.random.default_rng(1).permutation(n)
    model.fit(Xn[order], y[order], epochs=80, batch_size=8, verbose=0)
    return model


def eval_baseline(model, sequences, expected_idx):
    xs = np.stack([normalize_landmarks(np.asarray(seq)) for seq in sequences])
    preds = model.predict(xs, verbose=0)
    return int(np.sum(np.argmax(preds, axis=1) == expected_idx))


def main():
    QApplication(sys.argv)
    static_ds = json.load(open("data/datasets/dataset_static.json", encoding="utf-8-sig"))

    ctx = AppContext()
    screen = TranslateScreen(ctx)
    screen._load_classifiers()

    clf = screen.static_classifier
    h, o, l = pose(static_ds, "H", clf), pose(static_ds, "O", clf), pose(static_ds, "L", clf)
    hola_frames = (
        still(h, 40) + transition(h, o) + still(o, 40) + transition(o, l) + still(l, 40)
    )

    print("=== PARTE 1: pipeline en vivo (estaticas + compuertas) ===")
    resultado_hola = run(screen, ctx, hola_frames, "deletrear H-O-L con transiciones")

    print()
    print("=== PARTE 2: trayectoria (misma forma de mano, distinta ruta) ===")
    rng = np.random.default_rng(123)
    temp_dir = build_trajectory_workspace(static_ds, rng)
    try:
        result = TrainingService(PathService(temp_dir)).train_dynamic()
        print(f"entrenamiento temporal: trained={result.trained} muestras={result.samples} clases={result.classes}")
        paths = PathService(temp_dir)
        dyn_clf = DynamicSignClassifier(str(paths.dynamic_model_path), str(paths.dynamic_labels_path))

        shapes_l = shapes_by_letter(static_ds, "L")
        circles = [circle_seq(shapes_l[rng.integers(len(shapes_l))], rng) for _ in range(10)]
        lines = [line_seq(shapes_l[rng.integers(len(shapes_l))], rng) for _ in range(10)]
        pares = [(pose(static_ds, a, clf), pose(static_ds, b, clf)) for a, b in (("H", "O"), ("O", "L"), ("A", "B"), ("E", "F"), ("C", "D"))]
        transiciones = [transition_seq(a, b, rng) for a, b in pares for _ in range(2)]
        letras = ["A", "F", "I", "O", "E"]
        derivas = [drift_seq(shapes_by_letter(static_ds, letras[i % len(letras)])[0], rng) for i in range(10)]

        hits_circulo = eval_hits(dyn_clf, circles, "CIRCULO")
        hits_linea = eval_hits(dyn_clf, lines, "LINEA")
        hits_trans = eval_hits(dyn_clf, transiciones, "unknown")
        hits_deriva = eval_hits(dyn_clf, derivas, "unknown")
        print(f"circulo reconocido       : {hits_circulo}/10")
        print(f"linea reconocida         : {hits_linea}/10")
        print(f"transiciones -> NO_SENA  : {hits_trans}/10")
        print(f"deriva brazo -> NO_SENA  : {hits_deriva}/10")

        baseline = train_baseline_vieja_normalizacion(temp_dir)
        base_hits = eval_baseline(baseline, circles, 0) + eval_baseline(baseline, lines, 1)
        print(f"normalizacion VIEJA (por frame): {base_hits}/20 correctas")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print()
    print("=== VEREDICTO ===")
    print("H-O-L sigue perfecto      :", "OK" if resultado_hola == "HOL" else "FALLO", f"('{resultado_hola}')")
    print("trayectoria distinguible  :", "OK" if hits_circulo >= 8 and hits_linea >= 8 else "FALLO")
    print("NO_SENA filtra no-senas   :", "OK" if hits_trans >= 8 and hits_deriva >= 8 else "FALLO")
    print("antes era imposible       :", "SI" if base_hits <= 14 else "NO", f"({base_hits}/20, azar=10)")


if __name__ == "__main__":
    main()
