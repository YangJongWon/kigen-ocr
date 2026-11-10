# Entry point: keeps torch out of the spawned data workers (Windows).
if __name__ == "__main__":
    import train
    train.main()
