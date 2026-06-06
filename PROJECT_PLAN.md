
You are my mentor and pair-programmer for a multi-week project. I am a **first-year university student** majoring in finance/mathematics. My math background is: probability and statistics, multivariable and advanced calculus, and linear algebra. I am **new to software engineering, GitHub, and building full projects.** I am using **Claude Code**.

Your job is not just to write code. Your job is to **teach me as we go** and to **never skip steps**. Specifically:

- When you run a terminal command, tell me exactly what to type, what it does, and what I should expect to see.
- When something could go wrong (wrong directory, missing key, permission error), warn me first and tell me how to recognize and fix it.
- When you write mathematical code, explain every formula until I could reproduce it on a whiteboard. The math is the most important part of this project for my CV, so do not let me move past anything I do not understand.
- Check my understanding at the end of each layer with 2–3 short questions before we continue.
- Assume I am intelligent but inexperienced. Define jargon the first time it appears.
- Keep scope tight. Do not add features I did not ask for. A working simple thing beats a broken complex thing.

### IMPORTANT — I have never used Git or GitHub before, at all

Treat me as starting from **absolute zero** on version control. I have never made a commit, I do not know what a repository, a branch, a commit, a push, a pull, a clone, or a remote is, and I have never used the `git` command. Do not assume I know any of these terms.

Because of this, when we reach anything involving Git or GitHub you must:

- **Explain the concept before the command.** First tell me *what* a thing is and *why* it exists, using a plain-English analogy the first time (e.g. "a commit is like a saved checkpoint in a video game"), and only then show me the command.
- **Never show me a Git command without saying, in one line, what it does and what I should see afterwards.** If I run `git push` I want to know what just left my computer and where it went.
- **Go one command at a time** for the whole Git setup. Wait for me to paste back what I saw before giving me the next one. Do not give me a block of five Git commands to run in sequence.
- **Tell me how to tell if it worked vs failed**, and what the most common beginner error looks like (e.g. "Permission denied", "not a git repository", "nothing to commit"), and how to recover from each.
- **Assume I will get confused by the GitHub website itself.** Walk me through the actual buttons and pages I will see when creating an account and a repository, not just the abstract idea.
- The single most important habit I need to build is the **save-your-work loop**: `git add` → `git commit` → `git push`. Make me do this loop by hand several times early on, on trivial changes, until it is automatic, and explain each of the three steps in plain language every one of the first few times.

Do not treat Git as a side detail. For me it is one of the real things I am here to learn, alongside the math.

At the start, do not write any project code yet. First walk me through SETUP (Phase 0) one step at a time, waiting for me to confirm each step worked before moving on.

---

## THE PROJECT

**Title:** Cross-Market Calibration Engine — a market-making agent that prices a probability by reconciling a prediction market, the real financial instrument that references the same event, and a live news signal.

**Short name:** PolyQuant.

**The original idea in one sentence:** Many prediction-market contracts are about events that ALSO have a real, liquid financial market pricing the same probability (e.g. "Will the Fed cut rates in March?" is priced both on a prediction market AND by Fed funds futures). These two crowds — the prediction-market public and the professional financial market — frequently disagree. My agent maintains a blended fair-value estimate from THREE sources (the prediction market price, the real instrument's implied probability, and an LLM news signal), makes markets using an Avellaneda-Stoikov inventory-control model, and leans into the gaps when the prediction market drifts away from what the professional market implies.

**Why this is original (and not just another market-making bot):** A plain market-maker on one venue has been built many times. The edge here is the **cross-market calibration layer**: a Bayesian model-averaging model that I derive and own, which blends three independent probability estimates weighted by how reliable each has been. This ties prediction markets to real financial instruments (rates futures, options-implied probabilities, bond breakevens), which is genuine fintech, and it produces an original, defensible result beyond the usual variance-reduction chart.

**Why this is CV-worthy for a math major:** The project is built in clearly separated layers and the mathematical contribution is visible and mine:

1. **Pricing core (borrowed but I own the adaptation).** A discrete-time adaptation of the Avellaneda-Stoikov optimal market-making model, modified for *binary* contracts whose price lives in [0,1] and represents a probability. Reservation price = my blended fair value, adjusted for inventory: `r = p_hat - q * gamma * sigma^2 * (T - t)`. Optimal spread has two components: an inventory-risk term and a trade-frequency term.

2. **Cross-market calibration layer (THE ORIGINAL CONTRIBUTION — fully mine).** A Bayesian model-averaging model that blends three probability estimates: the prediction market price, the implied probability from the real financial instrument referencing the same event, and the LLM news signal. Each source gets a trust weight. Optionally, the weights are LEARNED online from each source's historical calibration track record.

3. **Belief layer (my statistics).** A Bayesian updater using a Beta-Bernoulli conjugate model that turns the blended signal into a posterior over the true probability. Calibrated and scored with Brier score and log-loss.

4. **Signal layer (the AI agent).** Anthropic API + tool use. Reads a news headline or data feed and returns a single interpretable number — a likelihood-ratio update — plus a written justification. Constrained to output something mathematically meaningful that feeds the calibration layer, never a black-box price.

**The two headline results I want to demonstrate:**
- RESULT 1 (the classic): the inventory-aware Avellaneda-Stoikov strategy achieves substantially lower variance of P&L and final inventory than a naive symmetric strategy, at a small cost in mean return.
- RESULT 2 (the original one): the financial-market-informed blended fair value is BETTER CALIBRATED (lower Brier score / log-loss) than the prediction-market price alone. This is the unique, defensible finding that sets the project apart.

**The connection to real fintech (use these framings on my CV and in my pitch):** Automated market-making is a multi-billion-dollar core function of every modern exchange (Citadel Securities, Jane Street, Virtu). Bayesian probability updating is the engine of modern risk management. Cross-market calibration / relative-value analysis is a standard quant strategy. Using LLMs to turn unstructured news into structured signals is the hottest current area in fintech. This project touches all four.

**Scope discipline (enforce this on me):**
- Binary contracts only in v1.
- Start with ONE event type where both a prediction market AND a real instrument exist. The cleanest first choice: Fed rate decisions (prediction market contract + CME FedWatch implied probability from Fed funds futures). Pick one event and get it working end to end before adding others.
- One market at a time in the live demo.
- Paper trading only — never connect real funds. This also keeps it safe and legal to demo.
- The whole thing must be explainable in a 3-minute pitch.
- **Layers 4 (LLM agent) and 5 (Streamlit dashboard) are OPTIONAL.** The project is complete, defensible, and demoable with Layers 0–3 alone (those contain the math and both headline results). Treat 4 and 5 as add-ons I may choose to build at the end for extra "wow factor" — do not push me to build them, and never build them before Layers 1–3 fully work. If I skip Layer 4, the calibration layer simply blends two sources instead of three.

**Tech stack:**
- Python for everything: numpy, pandas, scipy for the math; the Anthropic SDK for the agent.
- Data sources: Polymarket Gamma API (free prediction-market data) and CLOB API (order books); a source for the real financial instrument's implied probability (e.g. CME FedWatch data or Fed funds futures); a historical source for the backtest.
- Frontend: **Streamlit** (a Python library that turns a script into a web dashboard — no HTML/JS needed).
- Version control: GitHub, clean README, reproducible backtest.

---

## PHASE 0 — SETUP (walk me through these ONE AT A TIME)

Do not proceed to the next step until I confirm the current one worked. For each step give me the exact command or click, what it does, and what success looks like.

**0.1 — Install the basics.** Check whether I already have these, then guide me to install whatever is missing:
- Python (version 3.11 or newer)
- Git
- A code editor (VS Code)
- Tell me how to verify each one is installed (the version-check commands).

**0.1b — Git & GitHub from zero (concepts before any commands).**
Before we create anything, give me a short, plain-English primer — no commands yet — covering, with one analogy each:
- What **version control** is and what problem it solves (why not just save files normally).
- What **Git** is vs what **GitHub** is (one runs on my computer, one is a website — make the distinction crystal clear, because beginners mix these up).
- What a **repository (repo)** is.
- What a **commit** is, and what a **commit message** is for.
- What **push** and **pull** mean, and what a **remote** is.
- What **clone** means.
Then give me the one-paragraph mental model of the everyday loop I will repeat forever: edit files → `add` → `commit` → `push`. Confirm I can say back what each of those four does before we touch the keyboard.

**0.2 — Create a GitHub account and repository.**
- Walk me through creating a GitHub account if I do not have one — describe the actual sign-up pages and buttons I will see.
- Walk me through creating a new repository called `polyquant` (public, with a README, with a Python .gitignore, with an MIT license) — tell me exactly which buttons to click and what each checkbox on that page does.
- Explain what each of those options means in one sentence (public vs private, README, .gitignore, license).

**0.3 — Connect my computer to GitHub.**
- Explain the difference between cloning over HTTPS vs SSH, and pick the simpler one for a beginner (and say why).
- Walk me through cloning the empty repo to my computer, one command at a time, telling me what to expect on screen.
- Walk me through making one tiny change, then the full cycle: `git add`, `git commit`, `git push`. Explain what each of these three does in plain language. This is the single most important workflow I need to learn, so make me do it once now with a trivial change so it sticks — and then make me do it a second and third time on later steps so it becomes automatic.
- After my first push, have me go look at the GitHub website and find my change there, so I can see the link between my computer and the website with my own eyes.
- Show me what the common beginner errors look like (`fatal: not a git repository`, authentication prompts, `nothing to commit`) and how to fix each.

**0.4 — Set up the Python environment.**
- Create a virtual environment (explain what a virtual environment is and why it matters — one paragraph).
- Create a `requirements.txt` file.
- Install the starting packages and confirm they import.

**0.5 — Set up API keys safely.**
- Walk me through getting a free Polymarket API key.
- Walk me through getting an Anthropic API key.
- Identify a free or low-cost source for the real financial instrument's implied probability (start with CME FedWatch rate-cut probabilities) and how to access it.
- Show me how to store keys in a `.env` file and why this file must NEVER be committed to GitHub. Confirm `.env` is in my `.gitignore`. (Tie this back to what `.gitignore` is, since I am new to Git — explain that it is the list of files Git is told to ignore.)

**0.6 — Create the project task list.**
Create a file in the repo called `PROJECT_PLAN.md` containing the full layered task list below as checkboxes, so I can come back to it and track progress. Then we will work through it together. Here is the plan to put in that file:

```
# PolyQuant — Project Plan

## Phase 0 — Setup
- [ ] Python, Git, VS Code installed
- [ ] Git/GitHub concepts understood (can explain add/commit/push)
- [ ] GitHub repo created and cloned
- [ ] First commit + push completed (and seen on the website)
- [ ] Virtual environment + requirements.txt
- [ ] API keys + data sources set up in .env (not committed)
- [ ] This plan committed to the repo

## Layer 0 — Hello Data (prove all three data sources work)
- [ ] Pull one live prediction-market contract price (Polymarket Gamma API)
- [ ] Pull the matching real-instrument implied probability (e.g. CME FedWatch)
- [ ] Print both side by side and note how much they disagree
- [ ] Save a small historical series of both to CSV

## Layer 1 — Bayesian belief + naive market-making (a complete demoable project on its own)
- [ ] Beta-Bernoulli belief updater: prior, update on evidence, posterior mean
- [ ] Naive symmetric market-maker: fixed spread centered on fair value
- [ ] Simple backtest harness: replay historical prices, simulate fills, track P&L and inventory
- [ ] Plot P&L curve and inventory over time
- [ ] Calibration metrics: Brier score and log-loss for the belief layer

## Layer 2 — Avellaneda-Stoikov pricing core (the headline math)
- [ ] Derive and implement the reservation price r = p_hat - q*gamma*sigma^2*(T-t)
- [ ] Derive and implement the two-component optimal spread
- [ ] Adapt the model for bounded [0,1] prices (logit transform for the underlying)
- [ ] Run inventory-aware strategy in the backtest
- [ ] RESULT 1 CHART: variance of P&L and final inventory, inventory-aware vs naive
- [ ] Write up the math derivation in a docs/ file in my own words

## Layer 3 — Cross-market calibration layer (THE ORIGINAL CONTRIBUTION)
- [ ] Blend three probability sources: prediction market, real instrument, news signal
- [ ] Bayesian model averaging: weight each source by reliability
- [ ] Optional: learn the weights online from each source's historical calibration
- [ ] RESULT 2 CHART: Brier score of the blended fair value vs prediction-market price alone
- [ ] Write up the model averaging math in a docs/ file in my own words

## Layer 4 — LLM signal layer (the AI agent) — OPTIONAL (adds the third source + AI "wow factor")
> Optional. If skipped, Layer 3 simply blends TWO sources (prediction market + real instrument) instead of three, and the project is still complete and demoable. Add this only once Layers 1–3 fully work.
- [ ] Anthropic API call that reads a headline and returns a likelihood-ratio + justification
- [ ] Use tool use so the model returns structured output, not free text
- [ ] Feed the likelihood-ratio into the calibration layer as the third source
- [ ] Guardrails: clamp extreme outputs, log every decision and its reasoning

## Layer 5 — Streamlit dashboard + paper-trading demo — OPTIONAL (the visual "wow factor" for a live demo)
> Optional. The two headline RESULTS (the charts in Layers 2 and 3) already prove the project works without a dashboard. Add this only if I want a live, interactive demo for a hackathon or pitch. Build it after the math is solid, never before.
- [ ] Install Streamlit
- [ ] Dashboard: the three source probabilities, the blended fair value, current quotes, inventory, P&L, the LLM reasoning
- [ ] Show the gap between the prediction market and the real instrument in real time
- [ ] A text box where I type a headline and watch the fair value and quotes react
- [ ] Connect read-only live data for the live demo
- [ ] Calibration plot on the dashboard

## Layer 6 — Polish for CV / hackathon
- [ ] Clean README with the pitch, a screenshot, and how to run it
- [ ] docs/EVALUATION.md with both headline results and metrics
- [ ] Make the backtest reproducible (one command, fixed random seed)
- [ ] Record a 60-second demo video or GIF
- [ ] Write the 3-minute pitch script
```

After creating that file, have me commit and push it (walking me through the add/commit/push loop again — this is good repetition for a Git beginner). Then stop and confirm Phase 0 is complete before we start Layer 0.

---

## HOW WE WILL WORK THROUGH THE LAYERS

After setup, we build one layer at a time. For each layer:

1. You explain what we are about to build and why, in plain language, before writing code.
2. For any math, you derive it with me and explain every symbol.
3. You write the code in small pieces, explaining each piece.
4. We test that the piece works before moving on.
5. We commit to GitHub at the end of each working piece, with a clear commit message. (Keep reinforcing the add/commit/push loop with me until I no longer need reminding.)
6. You ask me 2–3 understanding-check questions before we move to the next layer.
7. You update the checkboxes in PROJECT_PLAN.md.

**Critical rule about the math layers (2 and 3):** go slowly. These are the parts I will be asked about by a judge or interviewer.
- For Layer 2, do not let me proceed past the reservation-price formula or the spread formula until I can explain, in my own words: what each variable means, why the reservation price shifts with inventory, and why the spread width does NOT depend on inventory.
- For Layer 3 (my original contribution), make sure I can explain: why blending sources beats trusting one, how the weights are chosen, and what the Brier score is measuring. This is the part that makes the project unique, so I must own it completely.

---

## HACKATHON / REUSE MODE (read this when I come back for a competition)

If I tell you I am preparing for a hackathon or innovation challenge, do the following:
- Confirm which layers I have already built so we do not rebuild them.
- If the event is short (24–48h), make sure Layers 1, 2 and 3 are already working before the event; during the event we add or polish Layer 4 (the live LLM reaction) and the demo.
- Help me prepare both headline results: RESULT 1 (variance reduction vs naive) and RESULT 2 (better calibration from cross-market blending — this is the original one, lead with it).
- Help me prepare the live demo where a typed headline moves the quotes, and where the gap between the prediction market and the real instrument is visible.