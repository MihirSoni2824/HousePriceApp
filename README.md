# 🦑🏠 SQUID GAME: HOUSE PRICE PREDICTION SURVIVAL 🏠🦑

## 🔴⚪🔺 WELCOME TO THE FINANCIAL SURVIVAL GAMES ⚪🔺🔴

*"456 data scientists entered. Only the most accurate predictions will survive..."*

```
┌─────────────────────────────────────────────────────────┐
│  PLAYER #001 - DATA SCIENTIST                           │
│  GAME: HOUSE PRICE PREDICTION CHALLENGE                 │
│  SURVIVAL STATUS: ████████░░ 80% COMPLETE               │
│  PRIZE MONEY: ₩45.6 BILLION (PERFECT PREDICTION)       │
└─────────────────────────────────────────────────────────┘
```

---

## 🎭 THE FRONT MAN'S WELCOME MESSAGE

*Adjusts golden mask and speaks through the intercom*

"Dear Players,

You have been selected to participate in the **House Price Prediction Games**. Like the children's games you once knew, the rules are simple, but the stakes are your financial future. 

Only those with the most accurate models will claim the ultimate prize. The weak will be... *eliminated* from the market."

**- The Front Man** 🎭

---

## 🟥🟨🟩 THE GAMES BREAKDOWN 🟩🟨🟥

### 🎮 **GAME 1: RED LIGHT, GREEN LIGHT** 
*Data Upload Challenge*

**🚦 THE RULES:**
- **GREEN LIGHT** 🟢: Upload your `house_prices.csv` successfully
- **RED LIGHT** 🔴: Any missing columns = **ELIMINATION**
- **MOVEMENT DETECTED**: Invalid data format = **GAME OVER**

**📋 Required Player Data:**
- **Size** (sq ft) - *Move only when this is numeric*
- **Location** (area) - *Stop if this contains nulls*
- **Number of Rooms** - *Freeze if non-integer values detected*
- **Price** ($) - *Ultimate target, any corruption = elimination*

**🏆 SURVIVAL REWARD:** Access to Honeycomb Challenge

---

### 🎮 **GAME 2: HONEYCOMB CHALLENGE**
*Data Preprocessing Survival*

**🍯 THE RULES:**
You must carefully extract clean data from the honeycomb without breaking it:

- **🔍 INSPECTION PHASE**: Detect missing values (cracks in the honeycomb)
- **⚙️ EXTRACTION PHASE**: Remove nulls with surgical precision
- **🎯 ENCODING PHASE**: One-hot encode locations without damaging the structure
- **📏 STANDARDIZATION**: Normalize numerical features to perfect proportions

**⏰ TIME LIMIT:** Process data before other players finish

**💀 ELIMINATION CONDITIONS:**
- Break the data structure during cleaning
- Leave missing values unhandled
- Corrupt numerical distributions

**🏆 SURVIVAL REWARD:** Entry to Tug of War Arena

---

### 🎮 **GAME 3: TUG OF WAR**
*Model Training Battle*

**🪢 THE RULES:**
Your Linear Regression model vs. the market's complexity:

- **⚖️ TEAM SPLIT**: 80% training data vs 20% testing data
- **💪 STRENGTH TEST**: Your model must pull the R² score above survival threshold
- **🎯 STRATEGY**: Lower RMSE = stronger rope grip
- **⚠️ DANGER ZONE**: R² below 0.5 = fall into the abyss

**📊 VICTORY CONDITIONS:**
- **🏆 LEGENDARY**: R² > 0.9 (Team Captain status)
- **🥇 STRONG SURVIVOR**: R² 0.7-0.9 (Safe passage)
- **🥈 BARELY ALIVE**: R² 0.5-0.7 (Hanging by thread)
- **💀 ELIMINATED**: R² < 0.5 (Fall to financial ruin)

**🏆 SURVIVAL REWARD:** Access to Final Game

---

### 🎮 **GAME 4: GLASS BRIDGE**
*Results Visualization Challenge*

**🌉 THE RULES:**
Navigate across the bridge of visualization choices:

**LEFT PATH - CORRELATION HEATMAP** 🔥
- **TEMPERED GLASS**: Deep insights into feature relationships
- **REGULAR GLASS**: Misleading correlations that shatter your understanding
- **SAFETY**: Save as JPG before stepping forward

**RIGHT PATH - ACTUAL VS PREDICTED SCATTER** 📈
- **TEMPERED GLASS**: Perfect predictions align with reality line
- **REGULAR GLASS**: Wild predictions scatter into chaos
- **SAFETY**: Document your model's truth before proceeding

**💡 SURVIVAL TIP:** Test each visualization carefully. One wrong interpretation leads to elimination.

**🏆 FINAL VICTORY:** Complete understanding of your model's true performance

---

## 🏢 INSTALLATION: ENTERING THE DORMITORY

### 🛏️ **Dormitory Setup Requirements**

Before you can sleep in the survival facility, prepare your environment:

```bash
# Standard Player Kit (Issue #456)
Python 3.8+ (Your survival weapon)
PySide6 (Dormitory interface access)
pandas (Data manipulation tools)
scikit-learn (Prediction algorithms)
matplotlib (Visualization equipment)
seaborn (Advanced plotting gear)
numpy (Mathematical foundation)
```

### 🚪 **Entry Protocol**

1. **Report to Facility**:
   ```bash
   git clone [facility-location-url]
   cd squid-game-house-prediction
   ```

2. **Receive Player Kit**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Enter the Games**:
   ```bash
   python main.py
   ```

---

## 🍽️ MEAL TIME: DATA FORMAT MENU

### 🥘 **Daily Rations (Required Data Columns)**

| Meal Component | Type | Nutrition Value | Player Example |
|----------------|------|-----------------|----------------|
| **Size Portion** | Numerical | Property sq ft energy | 1500, 2200, 3000 |
| **Location Soup** | Categorical | Area-based flavor | "Downtown", "Suburbs" |
| **Room Count Rice** | Numerical | Space density | 3, 4, 5 |
| **Price Protein** | Numerical | Target value nutrients | $250K, $450K, $750K |

### 🍱 **Sample Player Meal**:
```csv
Size,Location,Number of Rooms,Price
1200,Downtown,2,300000
1800,Suburbs,3,420000
2500,Beachfront,4,650000
```

**⚠️ WARNING:** Contaminated data = elimination from dining hall

---

## 🎨 FACILITY DESIGN: MINIMALIST AESTHETIC

### 🏢 **Architectural Theme**
Our facility combines the iconic Squid Game visual language:

- **🟩 GREEN TRACKSUITS**: Success indicators and progress bars
- **🟥 RED GUARDS**: Critical action buttons and warnings  
- **⬜ WHITE MINIMALISM**: Clean interface with surgical precision
- **⚫ VOID DARKNESS**: Background that represents elimination
- **🔺🔴⚪ GEOMETRIC SYMBOLS**: Navigation and status indicators

### 🎭 **Interactive Guard System**
- **👤 Pink Guards**: Basic functionality (hover effects)
- **🔺 Triangle Guards**: Advanced actions (model training)
- **⚪ Circle Guards**: Elimination warnings (error messages)
- **🟪 Square Guards**: Final authority (results validation)

---

## 🏗️ FACILITY BLUEPRINT

### 📁 **Complex Layout**

```
squid-game-house-prediction/
├── 🎭 main.py                 # Front Man's control center
├── 🏢 ui_main.py             # Dormitory interface system
├── ⚙️ preprocessing.py        # Data cleaning facility
├── 🤖 model.py               # Training arena algorithms
├── 🛠️ utils.py               # Guard utility functions
├── 📋 requirements.txt        # Player equipment list
└── 📊 house_prices.csv       # Game data (player provided)
```

### 🏢 **Facility Departments**

#### 🎭 **Control Room** (`ui_main.py`)
The Front Man's command center:
- **Surveillance Tabs**: Monitor player progress through games
- **Guard Communications**: Styling system with game aesthetics
- **Player Tracking**: Data flow management across challenges
- **Elimination Protocol**: Real-time visualization engine

#### ⚙️ **Data Processing Plant** (`preprocessing.py`)
Where data undergoes "preparation":
- `inspect_missing_and_summary()`: Player data intelligence gathering
- `normalize_and_encode()`: Data transformation for game compliance

#### 🤖 **Training Arena** (`model.py`)
Where models fight for survival:
- `split_and_train()`: Player vs market battle simulation
- `compute_metrics()`: Performance evaluation by guards
- `plot_correlation_heatmap()`: Visual intelligence reports
- `plot_actual_vs_predicted()`: Truth vs deception analysis

#### 🛠️ **Maintenance Crew** (`utils.py`)
Behind-the-scenes operations:
- `detect_outliers_iqr()`: Identify rule-breaking data points

---

## 🎯 PLAYER JOURNEY: GAME PROGRESSION

### 🚨 **Phase 1: Recruitment & Registration**
1. Initialize game interface via `python main.py`
2. Face the minimalist Squid Game UI
3. Notice security lockdown (only first game accessible)
4. **Guard Announcement**: "Welcome to the games, Player #001"

### 📁 **Phase 2: Game 1 - Red Light, Green Light**
1. Approach the **"🔍 Browse CSV"** terminal
2. Submit your data file for inspection
3. Survive the validation gauntlet:
   - Column requirement verification
   - Data integrity scan by triangle guards
   - Elimination protocol for invalid submissions
4. **Victory Bell**: Data preview unlocks, proceed to next game

### 🍯 **Phase 3: Game 2 - Honeycomb Challenge**
1. Analyze the honeycomb structure (missing values display)
2. Click **"⚙️ Normalize & Encode"** to begin extraction
3. Watch data undergo transformation:
   - Careful null value removal
   - Precise location encoding
   - Feature standardization process
4. **Success Chime**: Clean data preview, next game unlocked

### 🪢 **Phase 4: Game 3 - Tug of War**
1. Engage **"🚀 Train Model"** for the ultimate battle
2. Witness the 80/20 data division ceremony
3. Experience Linear Regression training montage
4. Receive judgment from the guards:
   - **R² Score**: Your survival rating
   - **RMSE**: Your elimination risk factor
5. **Victory Horn**: Acceptable performance grants final game access

### 🌉 **Phase 5: Game 4 - Glass Bridge**
1. **Correlation Heatmap Bridge**:
   - Auto-generation with spectral truth colors
   - Dark theme matching facility aesthetics
   - **"💾 Save Heatmap as JPG"** for victory documentation
2. **Prediction Accuracy Bridge**:
   - Reality vs model truth visualization
   - Perfect prediction line in elimination red
   - **"💾 Save Scatter Plot as JPG"** for survival proof

---

## 🏆 GUARD EVALUATION SYSTEM

### 🎯 **R² Score Judgment Protocol**
- **Range**: 0.0 to 1.0 (higher prevents elimination)
- **Guard Classifications**:
  - 0.9+ = **🏆 VIP STATUS** (90%+ market mastery)
  - 0.7-0.9 = **🟩 GREEN LIGHT SURVIVOR** (Strong performance)
  - 0.5-0.7 = **🟨 YELLOW WARNING** (Marginal survival)
  - Below 0.5 = **🔴 RED ELIMINATION** (Market failure)

### 🎯 **RMSE Elimination Threshold**
- **Currency**: Same denomination as your price data
- **Guard Assessment**:
  - Lower values = **Continued survival**
  - Example: RMSE $50,000 = Average error of $50K per prediction
  - Compare against your price range for elimination risk

---

## ⚠️ RULE VIOLATIONS & ELIMINATION

### 💀 **Common Elimination Scenarios**

#### **Game 1 Failures**:
```
🚨 VIOLATION: "Invalid CSV detected by triangle guards"
📋 CORRECTION: Verify columns: Size, Location, Number of Rooms, Price
```

#### **Game 2 Violations**:
```
🚨 VIOLATION: Data transformation sabotage detected
📋 CORRECTION: Ensure Size/Number of Rooms contain only numbers
```

#### **Game 3 Eliminations**:
```
🚨 VIOLATION: Insufficient training data for battle
📋 CORRECTION: Minimum 10+ clean data rows required for survival
```

#### **Game 4 Failures**:
```
🚨 VIOLATION: Visualization breach attempted
📋 CORRECTION: Complete model training before accessing results
```

### 🛡️ **Survival Optimization**
- **Large Datasets**: Strategic sampling for performance
- **Memory Conservation**: Close non-essential applications
- **Visual Performance**: Adjust rendering quality for older systems
- **Interface Speed**: Disable effects if experiencing lag

---

## 🎪 THE GAMES CONTINUE: FUTURE TOURNAMENTS

### 🎯 **Join the Development Squad**
Become a guard in our facility expansion:

1. **Infiltrate Repository** via fork
2. **Create Mission Branch**: `git checkout -b feature/new-game-mode`
3. **Implement Enhancement** following facility protocols
4. **Survival Testing** (no bugs survive our inspection)
5. **Submit Intelligence Report** via detailed pull request

### 💡 **Proposed New Games**
- **Marbles Game**: Feature selection challenges
- **Glass Bridge Extended**: Multi-model comparison
- **Final Dinner**: Hyperparameter optimization
- **VIP Experience**: Real-time prediction serving
- **Guard Training**: Advanced outlier detection
- **Facility Expansion**: Database integration protocols

---

## 🏛️ FACILITY REGULATIONS & COMPLIANCE

This facility operates under international data science ethics protocols. Remember: *"Knowledge is power, but prediction accuracy is survival."*

---

## 🎊 CONGRATULATIONS, SURVIVOR!

```
┌─────────────────────────────────────────────────────────┐
│              🏆 GAME COMPLETION CERTIFICATE 🏆          │
│                                                         │
│  PLAYER #001 has successfully survived all games       │
│  and demonstrated superior prediction capabilities      │
│                                                         │
│  ACHIEVEMENTS UNLOCKED:                                 │
│  ✅ Red Light Green Light Champion                      │
│  ✅ Honeycomb Extraction Master                         │
│  ✅ Tug of War Victory                                  │
│  ✅ Glass Bridge Navigator                              │
│                                                         │
│  FINAL PRIZE: ₩45.6 BILLION IN MARKET KNOWLEDGE       │
└─────────────────────────────────────────────────────────┘
```

### 🎭 **Front Man's Final Message**:

*"Congratulations, Player #001. You have proven that in the world of machine learning, survival belongs to those who understand both data and human nature. Your predictive model has earned you a place among the winners."*

### 🚀 **Post-Game Opportunities**:
- Deploy your winning model to production markets
- Train other players in advanced prediction techniques  
- Design new games for future tournaments
- Join the VIP observation deck for real-world applications

---

## 📞 FACILITY COMMUNICATIONS

Need assistance during your survival journey?

- **🚨 Emergency Protocols**: Report critical bugs via GitHub Issues
- **👥 Player Discussions**: Strategy sharing in GitHub Discussions  
- **📡 Facility Updates**: Star repository for new game announcements
- **🎭 Direct to Front Man**: Feature requests and facility improvements

---

## 🎮 **ENTER THE GAMES NOW**

*The Front Man awaits your participation...*

```bash
python main.py
```

```
┌─────────────────────────────────────────────────────────┐
│  "In this facility, you either predict accurately...    │
│   or your portfolio faces elimination.                  │
│   There is no middle ground in the market games."      │
│                                         - The Front Man │
└─────────────────────────────────────────────────────────┐
```

**🦑 Built with survival precision, powered by data science mastery! 🏠**

---

### 🔴⚪🔺 **Game Status: ACTIVE** ⚪🔺🔴

**May your predictions be accurate and your models survive! 🎯📊💰**

*Player #456, the games have begun...*
