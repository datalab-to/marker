import argparse
import os
import subprocess
import sys


def streamlit_app_cli(app_name: str = "streamlit_app.py"):
    argv = sys.argv[1:]
    if "--streamlit" in argv:
        argv = [a for a in argv if a != "--streamlit"]
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(cur_dir, app_name)
        cmd = [
            "streamlit",
            "run",
            app_path,
            "--server.fileWatcherType",
            "none",
            "--server.headless",
            "true",
        ]
        if argv:
            cmd += ["--"] + argv
        subprocess.run(cmd, env={**os.environ, "IN_STREAMLIT": "true"})
        return

    ap = argparse.ArgumentParser(
        description="Launch the Marker interactive web viewer (or pass --streamlit for the legacy Streamlit app)."
    )
    ap.add_argument("--port", type=int, default=8000, help="Port to run the viewer on")
    ap.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the viewer to"
    )
    ap.add_argument(
        "--dpi", type=int, default=150, help="Initial render resolution (DPI)"
    )
    ap.add_argument(
        "--max-pages", type=int, default=20, help="Initial page limit per document"
    )
    ap.add_argument(
        "--mode",
        choices=["auto", "no_ocr", "fast", "balanced", "vllm"],
        default="auto",
        help="Initial conversion mode",
    )
    args, _ = ap.parse_known_args(argv)

    try:
        import uvicorn
        from marker.scripts.server import app, viewer_engine

        if app is None:
            raise ImportError("FastAPI is not installed")

        viewer_engine.defaults.update(
            {
                "mode": args.mode,
                "dpi": args.dpi,
                "max_pages": args.max_pages,
            }
        )
        print(
            f"Marker Viewer & API Server running on http://{args.host}:{args.port}",
            flush=True,
        )
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        from http.server import ThreadingHTTPServer
        from marker.scripts.viewer import ViewerEngine, make_handler

        engine = ViewerEngine(
            defaults={
                "mode": args.mode,
                "force_ocr": False,
                "use_llm": False,
                "strip_existing_ocr": False,
                "keep_headers_footers": False,
                "dpi": args.dpi,
                "max_pages": args.max_pages,
            }
        )
        srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
        print(
            f"Marker Viewer running on http://{args.host}:{args.port}",
            flush=True,
        )
        srv.serve_forever()
