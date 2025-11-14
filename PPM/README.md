# PPM methodology

## **The Problem**
Imagine you're tracking someone's daily routine using GPS. You know where they were from Day 1-60, and you have some clues about Day 61-75. Your job: **predict their missing locations on Days 61-75**.

---

## **Phase 1: Data Loading & Preparation**

### **What we have:**
- **150,000 users** moving around a **200×200 grid** (each cell = 500m × 500m)
- **24 hourly timestamps** per day (0-23)
- **Days 1-60**: Complete historical data
- **Days 61-75**: Partial data (some locations revealed `✓`, some masked `?`)

### **The Split Strategy:**
```
Training Data = Days 1-60 (all users) + 70% of users from Days 61-75
Validation Data = 30% of users from Days 61-75
```

**Why this split?**
- Tests if model can predict for **users it has seen before** but in **new time periods**
- No data leakage: Validation users are completely held out

---

## **Phase 2: User Profiling (Building Memory)**

**Think of this as creating a "travel diary" for each user:**

### For each user, we store:

1. **Day Signatures** - What they did each day
   ```
   Day 5:  {8am: Home(50,60), 9am: Work(120,80), 5pm: Home(50,60)}
   Day 12: {8am: Home(50,60), 9am: Work(120,80), 6pm: Gym(100,90)}
   ```

2. **Hourly Patterns** - Where they usually are at each hour
   ```
   8am: Home(50,60) - 45 times, Café(55,65) - 5 times
   9am: Work(120,80) - 50 times
   ```

3. **Fallback Location** - Their most visited place overall
   ```
   Home(50,60) - visited 500 times
   ```

4. **POIs (Points of Interest)** - Popular locations per hour
   ```
   9am POIs: Office District(120,80), Train Station(90,70)
   7pm POIs: Mall(150,100), Restaurant Area(140,95)
   ```

**Memory Efficient:** Store only essential patterns, not raw data

---

## **Phase 3: Prediction Strategies**

### **The Core Idea:**
Given clues about today (Day 65), find the most similar past days, then use those days to predict missing timestamps.

---

### **Strategy 1: Cosine Basic (The Pattern Matcher)**

**Analogy:** Like finding similar movie-watching habits
- "You watched Action at 8pm and Comedy at 9pm"
- "Which of your past days match this pattern?"

**How it works:**
1. **Match timestamps:** Count how many times clue locations match historical days
   ```
   Day 65 clue:  {8am: (50,60), 12pm: (120,80)}
   
   Day 5:  {8am: (50,60) ✓, 12pm: (120,80) ✓}  → Score: 2/2 = 100%
   Day 12: {8am: (50,60) ✓, 12pm: (110,75) ✗}  → Score: 1/2 = 50%
   ```

2. **Pick Top 10 most similar days**

3. **Vote for each missing timestamp:**
   ```
   Need to predict 2pm:
   - Day 5 says: (130,85)
   - Day 12 says: (130,85)
   - Day 18 says: (130,85)
   - Day 22 says: (135,90)
   
   Winner: (130,85) with 3 votes
   ```

**Tie-breaker:** If tied, pick location that connects smoothly from 1pm

---

### **Strategy 2: Cosine POI (The Hotspot Hunter)**

**Analogy:** Like Yelp recommendations - popular places matter more

**Key difference from Strategy 1:**
- **Boosts POI matches:** If you visited Starbucks at 8am (a popular POI), finding days where you also visited Starbucks at 8am gets **2× score**
- **POI-to-POI partial credit:** Even different POIs get 0.5 points (both are popular spots)

**Example:**
```
Day 65 clue: {8am: Starbucks(50,60) [POI]}

Day 5:  {8am: Starbucks(50,60) [POI]}     → Score: 2.0 (POI exact match)
Day 12: {8am: Dunkin(55,65) [POI]}        → Score: 0.5 (both POIs)
Day 18: {8am: Home(45,55) [not POI]}      → Score: 0.0 (no match)
```

**Fallback:** If no clue has POIs, use top POIs for that hour
```
Need to predict 7pm, no matches found
→ Use "Mall(150,100)" (most popular 7pm location)
```

---

### **Strategy 3: Hybrid (The Smart Combiner)**

**Analogy:** Like asking both your personal diary AND TripAdvisor

**How it works:**
1. Get **Top 5 days** using POI-weighted matching
2. Get **Top 5 days** using basic matching
3. **Combine them** (remove duplicates, keep highest scores)
4. Vote using all 10 candidates

**Why it's better:**
- Captures both **personal habits** (basic) and **popular trends** (POI)
- More robust when one method fails

---

## **Phase 4: Ensemble (The Committee Decision)**

**Analogy:** Like having 3 weather forecasts and taking a weighted average

```
Prediction for Day 65, 2pm:

Strategy 1 (Basic):   (130,85)  [weight: 2.0]
Strategy 2 (POI):     (130,85)  [weight: 2.5]
Strategy 3 (Hybrid):  (135,90)  [weight: 2.0]

Votes:
- (130,85): 2.0 + 2.5 = 4.5 votes ← WINNER
- (135,90): 2.0 votes

Final prediction: (130,85)
```

**Weights explained:**
- **Strategy 2 (POI) = 2.5:** Highest because POIs are strong signals
- **Strategy 1 & 3 = 2.0:** Good baseline and hybrid
- **If all tied:** Use temporal continuity (connect smoothly from previous timestamp)

---

## 📏 **Phase 5: Evaluation (GeoBLEU Score)**

**GeoBLEU:** A metric that measures how well trajectories match (0 = terrible, 1 = perfect)

**It considers:**
1. **Spatial accuracy:** Are locations close to ground truth?
2. **Trajectory smoothness:** Does the path make sense?
3. **N-gram matching:** Considers sequences, not just individual points

**Example:**
```
Ground truth:    A → B → C → D
Prediction 1:    A → B → C → E  → GeoBLEU: 0.85 (good, just last point off)
Prediction 2:    A → X → Y → Z  → GeoBLEU: 0.15 (bad, only first point right)
```

---

## **Complete Flow**

```
1. Load Data
   ↓
2. Split: Train (Days 1-60 + 70% users 61-75) / Val (30% users 61-75)
   ↓
3. Build Profiles (day signatures, hourly patterns, POIs, fallbacks)
   ↓
4. For each prediction day:
   
                    ┌─────────────────────────────────────┐
                    │  Day 65 clue: {8am: A, 12pm: B}     │
                    │  Need: 2pm, 5pm, 8pm                │
                    └─────────────────────────────────────┘
                                    ↓
   ┌──────────────────────┬──────────────────────┬──────────────────────┐
   │  Strategy 1 (Basic)  │  Strategy 2 (POI)    │  Strategy 3 (Hybrid) │
   │  Score: Exact match  │  Score: POI boost    │  Score: Combined     │
   │  Top 10 days         │  Top 10 days         │  Top 10 days         │
   │  Vote → Predict      │  Vote → Predict      │  Vote → Predict      │
   └──────────────────────┴──────────────────────┴──────────────────────┘
           ↓                       ↓                       ↓
           
                Final predictions: {2pm: C, 5pm: D, 8pm: E}
                        ↓
5. Compare with ground truth → Calculate GeoBLEU
```

---

## **Key Innovations**

✅ **Proper validation:** User-based split prevents leakage  
✅ **Multiple strategies:** Captures different patterns  
✅ **POI awareness:** Leverages popular locations  
✅ **Robust fallbacks:** Never fails, always predicts something  
✅ **Temporal continuity:** Predictions connect smoothly  
✅ **Weighted ensemble:** Best strategies get more influence  

**Final output:** CSV with predictions for target users across Days 61-75!