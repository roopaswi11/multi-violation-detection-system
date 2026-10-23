import requests
import json
import os

url = "http://127.0.0.1:5000/upload"
test_image = r"uploads/24361fe8_Screenshot_2026-10-12_183619.png"

if not os.path.exists(test_image):
    # Find any sample image in uploads or dataset
    import glob
    candidates = glob.glob("uploads/*.png") + glob.glob("uploads/*.jpg")
    if candidates:
        test_image = candidates[0]

print(f"Uploading {test_image} to {url}...")
with open(test_image, "rb") as f:
    files = {"file": (os.path.basename(test_image), f, "image/png")}
    data = {"direction": "two_way", "lane": "all"}
    headers = {"X-Requested-With": "XMLHttpRequest"}
    resp = requests.post(url, files=files, data=data, headers=headers)

print(f"Response status: {resp.status_code}")
if resp.status_code == 200:
    res = resp.json()
    print("Success:", res.get("success"))
    print("Job ID:", res.get("job_id"))
    print("Violations detected:", len(res.get("violations", [])))
    for idx, v in enumerate(res.get("violations", [])):
        print(f"  [{idx+1}] Type: {v.get('violation_type')}, Status: {v.get('review_status')}, Rider: {v.get('rider_id')}, Conf: {v.get('confidence')}")

    # Now verify results page HTML rendering
    results_url = f"http://127.0.0.1:5000/results/{res.get('job_id')}"
    r_html = requests.get(results_url)
    print(f"Results page status: {r_html.status_code}")
    assert "Confirmed" in r_html.text or "Violations" in r_html.text
    print("Results page loaded and verified successfully!")
else:
    print(resp.text[:500])
