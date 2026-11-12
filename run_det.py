# Entry point for detect.py: keeps torch out of the spawned data workers (Windows).
if __name__ == "__main__":
    import detect
    detect.main()
