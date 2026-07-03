# TODO - Make Webpage Run Without Any Error

## Step 1: Verify Flask app startup and routes
- [ ] Fix any startup-time crashes (model loading, TensorFlow availability, path issues)
- [ ] Verify routes render: `/`, `/about`, `/how-it-works`, `/upload`, `/result`, error pages

## Step 2: Fix missing asset path issues
- [ ] Ensure static assets referenced in templates resolve correctly (e.g., logo paths)

## Step 3: Fix prediction path errors
- [ ] Ensure upload folder and image serving work for uploaded files
- [ ] Ensure `predict_disease()` handles missing model/TensorFlow gracefully

## Step 4: Add robust error handling
- [ ] Return friendly errors instead of server 500s when prediction fails

## Step 5: Smoke test
- [ ] Run server and fetch each page; upload a test image; confirm `/result` renders

