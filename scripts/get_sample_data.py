import os
import requests

DATA_DIRS = {
    "data/test_sunny": [
        "https://images.unsplash.com/photo-1625232789174-8d9904944b74?w=600&q=80",
        "https://images.unsplash.com/photo-1473187983305-f615310e7daa?w=600&q=80",
        "https://images.unsplash.com/photo-1595180634674-60911762c974?w=600&q=80",
    ],
    "data/test_rain": [
        "https://images.unsplash.com/photo-1512592067672-c584d4365758?w=600&q=80",
        "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=600&q=80",
        "https://images.unsplash.com/photo-1505536560938-168d6a85536d?w=600&q=80",
    ],
    "data/test_night": [
        "https://images.unsplash.com/photo-1520113412646-042c5b364407?w=600&q=80",
        "https://images.unsplash.com/photo-1498661694102-0a3793ed4d83?w=600&q=80",
        "https://images.unsplash.com/photo-1519750292352-c9fc17322ed7?w=600&q=80",
    ]
}

def download_images():
    print("🚀 Starting Automated Data Collection...")
    for folder, urls in DATA_DIRS.items():
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    with open(f"{folder}/sample_{i}.jpg", 'wb') as f:
                        f.write(response.content)
            except:
                pass
    print("✅ Data Downloaded.")

if __name__ == "__main__":
    download_images()
