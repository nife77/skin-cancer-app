"""Run this script with Streamlit (from project root):

  streamlit run streamlit_app.py

This is the safe top-level entrypoint that imports from the `app` package.
"""
from app.ui import main


if __name__ == "__main__":
    # For direct execution (not typical when using `streamlit run`), allow running
    main()
