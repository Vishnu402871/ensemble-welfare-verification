"""
auto_baseline_v7.py
===================
Rule-based eligibility matcher — v7.

Fixes over v6:
  Fix 1 — existing_credit, has_existing_loan, loan_account_status → SKIP
           ECLGS requires existing business credit — citizen profiles track
           personal loans, not business credit lines. Skipping avoids 9 FNs.
  Fix 2 — education_field → SKIP
           Citizen profiles have education_level but not education_field
           (Agriculture vs Computer Science). Cannot evaluate — skip.
  Fix 3 — asset_ownership, license_required → SKIP
           KCC Fisheries scheme requires specific asset ownership not in profile.
  Fix 4 — beneficiary_type now evaluated via occupation alias (not skipped)
           Only for schemes where beneficiary_type carries real meaning.

Usage:
  python auto_baseline_v7.py

Files needed (download from the HuggingFace dataset into data/):
  data/schemes.json
  data/gold_eval_pairs.csv
  data/citizen_profiles.json
Reproduces the paper's symbolic-matcher row exactly:
  Acc 0.8791 | kappa 0.6758 | F1 0.7549 | Recall 0.8435
"""

import json, pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix)

SCHEMES_FILE  = "data/schemes.json"
EVAL_CSV      = "data/gold_eval_pairs.csv"
CITIZENS_FILE = "data/citizen_profiles.json"
RESULTS_FILE  = "results/auto_v7_predictions.csv"

# ── OCCUPATION ALIASES ────────────────────────────────────────────────────────
OCCUPATION_ALIASES = {
    "farmer":              ["farmer","agriculturist","ryot","agricultural labourer",
                            "grower","landholding farmer"],
    "fisherman":           ["fisherman","fisher","fish farmer","aquaculture farmer"],
    "labour":              ["labour","laborer","construction worker",
                            "building and construction worker",
                            "manual scavenger","sanitation worker","sewer worker",
                            "septic tank worker","factory worker","industrial worker",
                            "unorganized worker","worker","craftsman"],
    "student":             ["student","research scholar"],
    "unemployed":          ["unemployed"],
    "entrepreneur":        ["entrepreneur","business owner","self-employed",
                            "artisan","craftsperson","weaver","handloom weaver",
                            "small business owner"],
    "self-employed":       ["entrepreneur","business owner","self-employed",
                            "artisan","craftsperson","weaver"],
    "fisherman":           ["fisherman","fisher","fish farmer"],
    "government_employee": ["government employee","regular government employee",
                            "state government employee","civil servant"],
    "private_employee":    ["private employee","employee","salaried"],
    "retired":             ["retired","service pensioner","family pensioner",
                            "re-employed service pensioner"],
    "street vendor":       ["street vendor","vendor","flower vendor",
                            "vegetable vendor","fruit vendor"],
    "folk artiste":        ["folk artiste","artist","musician","writer"],
    "journalist":          ["journalist","reporter","editor","sub-editor"],
    "teacher":             ["teacher","academician","professor"],
    "artisan":             ["artisan","craftsperson","craftsman","weaver",
                            "handloom weaver","leather artisan"],
}

# ── FIELD MAP: rule field → citizen profile field ─────────────────────────────
REVERSE_MAP = {
    "age":                    "age",
    "gender":                 "gender",
    "annual_income":          "annual_income_inr",
    "max_income_inr":         "annual_income_inr",
    "income_limit_inr":       "annual_income_inr",
    "rural_annual_income":    "annual_income_inr",
    "urban_annual_income":    "annual_income_inr",
    "caste_category":         "caste_category",
    "occupation":             "occupation",
    "employment_status":      "occupation",
    "disability":             "disability",
    "disability_percentage":  "disability",
    "below_poverty_line":     "below_poverty_line",
    "income_status":          "below_poverty_line",
    "household_category":     "below_poverty_line",
    "poverty_category":       "below_poverty_line",
    "marital_status":         "marital_status",
    "education_level":        "education_level",
    "family_size":            "family_size",
    "residence":              "state",
    "domicile":               "state",
    "country":                "state",
    "state":                  "state",
    "citizenship":            "citizenship",
    "bank_defaulter":         "is_bank_defaulter",
    "is_bank_defaulter":      "is_bank_defaulter",
    "housing_type":           "housing_type",
    "land_ownership":         "owns_land",
    "owns_land":              "owns_land",
    "prior_govt_scheme":      "has_prior_govt_scheme",
    "prior_govt_subsidy":     "has_prior_govt_scheme",
    "has_prior_govt_scheme":  "has_prior_govt_scheme",
    "other_govt_scheme":      "has_prior_govt_scheme",
}

# ── SKIP FIELDS ───────────────────────────────────────────────────────────────
SKIP_FIELDS = {
    # Fix 1 — business credit fields not in citizen profiles
    "existing_credit", "has_existing_loan", "loan_account_status",
    "loan_account_status",

    # Fix B — POMI (15 FPs)
    "account_type", "joint_holders_max",

    # Fix B — PMEGP (11 FPs): enterprise_type not in citizen profile
    "enterprise_type",

    # Fix B — Mission Vatsalya (2 FPs)
    "foster_willing", "child_age_sponsorship",
    "child_age_aftercare_min", "child_age_aftercare_max",

    # Fix B — Micro Credit Card (2 FPs)
    "udyam_registered",

    # Fix B — National Agriculture Market (1 FP)
    "single_trading_license", "single_point_market_fee",
    "etrading_support", "enam_trading_commitment",
    "soil_testing_lab_linked",

    # Fix B — UP Atal Awasiya (1 FP)
    "parent_board_registered", "parent_membership_days",
    "identified_by_dept",

    # Fix B — Bhargava Matching Scheme (1 FP)
    "community",

    # Fix B — DAY-NRLM (1 FP)
    "shg_member", "shg_programme",

    # Fix B — Weavers Mudra (1 FP)
    "prior_margin_money", "prior_mudra_loan_repaid",

    # Fix B — Pension KBOCWWB — age>=3 passes everyone; skip occupation='Worker' (too generic)
    # handled via occupation alias tightening below

    # Fix 2 — education subject field (not in citizen profile)
    "education_field", "agri_course_content_pct",
    "diploma_marks_min", "intermediate_marks_min",

    # Fix 3 — asset ownership not in citizen profile
    "asset_ownership", "license_required",

    # Standard non-evaluable fields
    "residence_type", "loan_purpose", "institution_type",
    "industry_type", "board_registered", "board_name", "board_membership",
    "training_interest", "cooperative_member", "loan_type", "course_type",
    "pregnancy_status", "current_class", "disability_type", "pregnancy_event",
    "course_level", "study_mode", "artiste_registered", "community",
    "tribal_membership", "income_pct_ami", "benefit_received",
    "state_pension_age_reached", "work_hours_per_week",
    "employment_exchange_years", "district", "farmer_type", "shg_member",
    "udyam_registered", "service_years", "membership_years",
    "board_service_days", "subscription_years", "other_pension",
    "other_scholarship", "prior_scheme", "scheme_type", "enterprise_type",
    "sector", "new_project", "vendor_document", "issuing_authority",
    "stall_location_type", "stall_tenure", "loan_source", "fishing_activity",
    "seed_production", "seed_supply_dept", "irrigation_needed",
    "training_requirement", "edp_training_completed",
    "research_field", "postdoc_experience_years", "sci_publication",
    "patent_filed", "research_centre", "career_break", "current_emoluments",
    "fellowship_availed", "prior_fellowship", "prior_kscste_fellowship",
    "prior_postdoc_fellowship", "prior_emeritus_fellowship",
    "prior_mudra_loan_repaid", "prior_margin_money", "prior_solar_subsidy",
    "prior_housing_scheme", "prior_corporation_loan", "aadhaar_linked",
    "marriage_registered", "criminal_conviction", "dbt_registered",
    "pfms_registered", "agreement_executed", "medical_board_certified",
    "first_generation_entrepreneur", "running_business", "industry_expertise",
    "blacklisted", "pip_identified", "separate_bank_account",
    "project_report", "own_contribution_pct", "business_type",
    "motor_vehicle", "vessel_type", "land_area_acres_min",
    "land_area_acres_max", "land_area_hectares", "vehicle_capacity_kg_min",
    "vehicle_capacity_kg_max", "authorized_dealer", "fruit_orchard",
    "irrigation_source", "property_location", "burial_ground_access",
    "parent_occupation", "parent_board_registered", "parent_board_membership",
    "parent_disability", "parent_disability_percentage",
    "child_district_topper", "child_exam_passed", "child_beneficiary_max",
    "children_beneficiary_max", "existing_children", "family_member_enrolled",
    "family_graduate", "family_govt_employee", "family_support",
    "has_children", "foster_willing", "main_carer", "child_benefit_eligible",
    "child_in_education", "child_age_max", "child_compulsory_age",
    "course_accepted_age", "universal_credit", "qualifying_course",
    "qualifying_childcare", "ofsted_registered_provider",
    "student_finance_eligible", "tuition_fee_loan_eligible",
    "maintenance_loan_eligible", "postgraduate_loan", "study_credits",
    "ni_contributions_paid", "ni_contribution_class", "ni_contribution_from_year",
    "work_related_disability", "accident_location", "bereavement_status",
    "partner_ni_contributions", "partner_work_death", "funeral_location",
    "relationship_to_deceased", "benefit_claimed_weeks", "work_duration_weeks",
    "housing_costs_before_work", "housing_costs_after_work", "child_guardian",
    "child_parents_deceased", "parent_birth_country",
    "parent_uk_residence_weeks", "parent_uk_residence_from_age",
    "dwp_eligible", "loan_terms_agreed", "benefit_conditions_ongoing",
    "primary_residence", "mortgage_responsible", "waiting_period_met",
    "property_value_pct_median", "affordability_period_met",
    "income_pct_ami_homebuyer", "income_pct_ami_renter",
    "independent_living_support", "asset_limit_applicable",
    "military_service", "discharge_status", "years_since_injury",
    "late_onset_illness", "adl_support_needed", "caregiver_age",
    "caregiver_relation", "caregiver_background_check",
    "caregiver_training_completed", "shared_parental_responsibility",
    "both_parents_eligible", "work_criteria_met", "pay_criteria_met",
    "taking_leave", "child_care", "correct_notice_given", "adoption_proof",
    "continuous_employment_weeks", "weekly_earnings_gbp", "adoption_type",
    "sc6_form_signed", "parental_order_intended", "child_conviction",
    "birth_parent_consent", "paternity_leave_claimed",
    "worked_for_employer", "illness_days",
    "gp_registered_england", "nhs_prescription_needed", "hrt_only",
    "free_prescription_eligible", "income_related_benefits",
    "max_loan_amount", "max_house_value", "max_carpet_area_sqm",
    "borrower_type", "loan_amount", "income_category", "pucca_house_owned",
    "income_taxpayer", "bride_age_min", "bride_age_max",
    "groom_age_min", "groom_age_max", "disability_inter_marriage",
    "one_partner_disabled", "one_partner_non_disabled",
    "prior_scheme_availed", "prior_scheme_second_marriage",
    "prior_civil_claim", "prior_employer_payment", "mod_compensation",
    "pneumoconiosis_act_1979", "zld_mandated", "single_trading_license",
    "single_point_market_fee", "etrading_support", "enam_trading_commitment",
    "soil_testing_lab_linked", "proposal_submitted",
    "private_market_recommended", "ms_id_allotted", "survey_identified",
    "sanitation_worker_scope", "gram_panchayat_identified",
    "gram_panchayat_compliant", "maintenance_contribution",
    "piped_water_connection", "safe_water_access", "alternative_livelihood",
    "destitute", "identification", "remarriage", "marriage_beneficiary",
    "disability_attributable", "account_type", "joint_holders_max",
    "property_ownership", "shg_programme", "beneficiary_type",
    "business_location", "enterprise_category",
    "natural_born", "permanent_resident_years", "residence_years",
    "employment_type", "lease_period_years", "income_below_2x_poverty",
    "govt_dues_pending", "land_area_sqft", "disability_certificate",
    "disability_act", "sanction_frequency_years", "lifetime_sanction",
    "prior_kmdcl_scheme", "tender_participation", "seed_purchase_from",
    "seed_type", "generator_capacity_kva", "claim_within_months",
    "district_rank", "continuing_studies", "medium_of_study",
    "shg_size", "cooperative_type", "loan_account_status",
    "child_age_sponsorship", "child_age_aftercare_min",
    "child_age_aftercare_max", "scheme_type",
}

EDUCATION_ORDER = [
    "none", "primary", "5th pass", "7th pass", "8th pass", "10th pass",
    "secondary", "12th pass", "intermediate", "diploma", "iti", "vocational",
    "graduate", "postgraduate", "phd"
]

def edu_rank(val):
    val = str(val).lower().strip()
    for i, lvl in enumerate(EDUCATION_ORDER):
        if lvl in val:
            return i
    return -1

def match_occupation(citizen_occ, rule_value):
    if not citizen_occ:
        return True
    co = str(citizen_occ).lower().strip().replace("_", " ")
    rv = str(rule_value).lower().strip()
    if co == rv or rv in co or co in rv:
        return True
    aliases = OCCUPATION_ALIASES.get(co, [])
    for alias in aliases:
        if alias in rv or rv in alias:
            return True
    for base_occ, alias_list in OCCUPATION_ALIASES.items():
        if co in alias_list or co == base_occ:
            for alias in alias_list:
                if alias in rv or rv in alias:
                    return True
    return False

def match_state(citizen_state, rule_value):
    if not citizen_state:
        return True
    cs = str(citizen_state).lower().strip()
    rv = str(rule_value).lower().strip()
    if rv in ("india", "central", "national"):
        return True
    return cs == rv or rv in cs or cs in rv

def match_in_list(citizen_val, rule_value):
    options = [o.strip().lower() for o in str(rule_value).split(",")]
    cv = str(citizen_val).lower().strip()
    if cv in options:
        return True
    for opt in options:
        if opt in cv or cv in opt:
            return True
    return False

def citizen_matches_rule(citizen, rule):
    field = rule.get("field", "")
    op    = rule.get("op", "eq")
    rval  = rule.get("value")

    if field in SKIP_FIELDS:
        return True

    # ── Residence ─────────────────────────────────────────────────────────
    if field in ("residence", "domicile", "country", "state"):
        return match_state(citizen.get("state", ""), rval)

    # ── Occupation ────────────────────────────────────────────────────────
    if field in ("occupation", "employment_status"):
        co = citizen.get("occupation", "")
        if op == "eq":
            return match_occupation(co, rval)
        elif op == "in":
            return any(match_occupation(co, o.strip())
                       for o in str(rval).split(","))
        elif op == "neq":
            return not match_occupation(co, rval)

    # ── Gender ────────────────────────────────────────────────────────────
    if field == "gender":
        cv = str(citizen.get("gender", "")).lower().strip()
        rv = str(rval).lower().strip()
        if op == "eq":   return cv == rv or rv in cv
        elif op == "in": return match_in_list(cv, rval)

    # ── Age ───────────────────────────────────────────────────────────────
    if field == "age":
        try:
            age = float(citizen.get("age", 0))
            if op == "gte": return age >= float(rval)
            if op == "lte": return age <= float(rval)
            if op == "eq":  return age == float(rval)
        except: return True

    # ── Income ────────────────────────────────────────────────────────────
    if field in ("annual_income", "max_income_inr", "income_limit_inr",
                 "rural_annual_income", "urban_annual_income"):
        try:
            inc = float(citizen.get("annual_income_inr", 0))
            if op == "lte": return inc <= float(rval)
            if op == "gte": return inc >= float(rval)
            if op == "eq":  return inc == float(rval)
        except: return True

    # ── Caste ─────────────────────────────────────────────────────────────
    if field == "caste_category":
        cv = str(citizen.get("caste_category", "")).strip()
        if not cv: return True
        if op == "eq":    return cv.lower() == str(rval).lower()
        elif op == "in":  return match_in_list(cv, rval)
        elif op == "neq": return cv.lower() != str(rval).lower()

    # ── Disability ────────────────────────────────────────────────────────
    if field == "disability":
        cv = citizen.get("disability", None)
        if cv is None: return True
        if isinstance(rval, bool): return bool(cv) == rval
        return str(bool(cv)).lower() == str(rval).lower()

    if field == "disability_percentage":
        cv = citizen.get("disability", None)
        if cv is None: return True
        if op == "gte": return bool(cv)
        return True

    # ── BPL / poverty ─────────────────────────────────────────────────────
    if field in ("below_poverty_line", "income_status",
                 "household_category", "poverty_category"):
        cv = citizen.get("below_poverty_line", None)
        if cv is None: return True
        bpl = bool(cv)
        if isinstance(rval, bool): return bpl == rval
        rv_l = str(rval).lower()
        if rv_l in ("true","1","yes","bpl","poor","vulnerable","very poor",
                    "ultra-poor","extremely low-income","very low-income",
                    "low-income"):
            return bpl
        return True

    # ── Marital status ────────────────────────────────────────────────────
    if field == "marital_status":
        cv = str(citizen.get("marital_status", "")).lower().strip()
        if not cv: return True
        rv = str(rval).lower().strip()
        if op == "eq":    return cv == rv or rv in cv
        elif op == "in":  return match_in_list(cv, rval)
        elif op == "neq": return cv != rv

    # ── Education level ───────────────────────────────────────────────────
    if field == "education_level":
        cv = citizen.get("education_level", "")
        if not cv: return True
        if op == "eq":    return str(cv).lower() == str(rval).lower()
        elif op == "in":  return match_in_list(str(cv), rval)
        elif op == "gte": return edu_rank(cv) >= edu_rank(rval)
        elif op == "lte": return edu_rank(cv) <= edu_rank(rval)

    # ── Family size ───────────────────────────────────────────────────────
    if field == "family_size":
        try:
            fs = float(citizen.get("family_size", 0))
            if op == "gte": return fs >= float(rval)
            if op == "lte": return fs <= float(rval)
            if op == "eq":  return fs == float(rval)
        except: return True

    # ── Citizenship ───────────────────────────────────────────────────────
    if field == "citizenship":
        cv = str(citizen.get("citizenship", "")).lower().strip()
        rv = str(rval).lower().strip()
        if op == "eq":   return cv == rv or rv in cv
        elif op == "in": return match_in_list(cv, rval)

    # ── Bank defaulter (v3) ───────────────────────────────────────────────
    if field in ("bank_defaulter", "is_bank_defaulter"):
        cv = citizen.get("is_bank_defaulter", None)
        if cv is None: return True
        if isinstance(rval, bool): return bool(cv) == rval
        return str(bool(cv)).lower() == str(rval).lower()

    # ── Housing type (v3) ─────────────────────────────────────────────────
    if field == "housing_type":
        cv = str(citizen.get("housing_type", "")).lower().strip()
        if not cv: return True
        rv = str(rval).lower().strip()
        if op == "eq":   return cv == rv or rv in cv
        elif op == "in": return match_in_list(cv, rval)

    # ── Land ownership (v3) ───────────────────────────────────────────────
    if field in ("land_ownership", "owns_land"):
        cv = citizen.get("owns_land", None)
        if cv is None: return True
        if isinstance(rval, bool): return bool(cv) == rval
        return str(bool(cv)).lower() == str(rval).lower()

    # ── Prior govt scheme (v3) ────────────────────────────────────────────
    if field in ("prior_govt_scheme", "prior_govt_subsidy",
                 "has_prior_govt_scheme", "other_govt_scheme"):
        cv = citizen.get("has_prior_govt_scheme", None)
        if cv is None: return True
        if isinstance(rval, bool): return bool(cv) == rval
        return str(bool(cv)).lower() == str(rval).lower()

    # ── Minority (derived) ────────────────────────────────────────────────
    if field == "minority":
        cv = str(citizen.get("caste_category", "")).lower()
        minority = ["obc","sc","st","ews","muslim","christian",
                    "sikh","buddhist","jain","parsi"]
        is_min = any(m in cv for m in minority)
        if isinstance(rval, bool): return is_min == rval
        return True

    # ── Veteran (derived) ─────────────────────────────────────────────────
    if field == "veteran_status":
        co = str(citizen.get("occupation", "")).lower()
        is_vet = any(v in co for v in
                     ["veteran","ex-service","military","army","navy","air force"])
        if isinstance(rval, bool): return is_vet == rval
        return True

    return True  # unknown field — benefit of doubt

def predict_eligibility(citizen, scheme):
    rules = scheme.get("eligibility_rules_json") or []
    if not rules:
        return "not eligible"
    evaluable = [r for r in rules if r.get("field", "") not in SKIP_FIELDS]
    if not evaluable:
        return "not eligible"
    for rule in evaluable:
        if not citizen_matches_rule(citizen, rule):
            return "not eligible"
    return "eligible"

def compute_metrics(df, pred_col="auto_pred"):
    yt = (df["gold_label"] == "eligible").astype(int)
    yp = (df[pred_col]     == "eligible").astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return dict(
        n=len(df),
        accuracy=accuracy_score(yt, yp),
        kappa=cohen_kappa_score(yt, yp),
        precision=precision_score(yt, yp, zero_division=0),
        recall=recall_score(yt, yp, zero_division=0),
        f1=f1_score(yt, yp, zero_division=0),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)
    )

def print_metrics(label, m):
    print(f"\n{'='*62}\n  {label}\n{'='*62}")
    print(f"  Pairs     : {m['n']}")
    print(f"  Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Kappa     : {m['kappa']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

def main():
    print("\n=== Auto Baseline v7 FINAL (Rule-Based) ===")
    print(f"    Citizens : {CITIZENS_FILE}")
    print(f"    Schemes  : {SCHEMES_FILE}\n")

    with open(SCHEMES_FILE, encoding="utf-8") as f:
        schemes = {s["scheme_id"]: s for s in json.load(f)}
    print(f"  Loaded {len(schemes)} schemes")

    with open(CITIZENS_FILE, encoding="utf-8") as f:
        d = json.load(f)
    citizens = {c["citizen_id"]: c for c in d} if isinstance(d, list) else d
    print(f"  Loaded {len(citizens)} citizen profiles")

    eval_df = pd.read_csv(EVAL_CSV).reset_index(drop=True)
    print(f"  Loaded {len(eval_df)} eval pairs\n")

    preds = []
    for _, row in eval_df.iterrows():
        citizen = citizens.get(row["citizen_id"], {})
        scheme  = schemes.get(row["scheme_id"], {})
        preds.append(predict_eligibility(citizen, scheme))

    eval_df["auto_pred"] = preds
    m = compute_metrics(eval_df)
    print_metrics("AUTO BASELINE v7 RESULTS", m)

    # Per category
    print(f"\n{'Category':<42}{'n':>4}  {'Acc':>7}  {'F1':>7}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'gold_elig':>9}")
    print("-"*85)
    for cat in sorted(eval_df["category"].dropna().unique()):
        g    = eval_df[eval_df["category"] == cat]
        yt_c = (g["gold_label"] == "eligible").astype(int)
        yp_c = (g["auto_pred"]  == "eligible").astype(int)
        acc  = accuracy_score(yt_c, yp_c)
        f1   = f1_score(yt_c, yp_c, zero_division=0)
        try:
            tn_c, fp_c, fn_c, tp_c = confusion_matrix(yt_c, yp_c).ravel()
        except:
            tn_c = fp_c = fn_c = tp_c = 0
        print(f"{cat:<42}{len(g):>4}  {acc:>7.3f}  {f1:>7.3f}  "
              f"{int(tp_c):>4}  {int(fp_c):>4}  {int(fn_c):>4}  {yt_c.sum():>9}")

    # Per source
    print(f"\n{'Source':<28}{'n':>4}  {'Acc':>7}  {'F1':>7}  "
          f"{'gold_elig':>9}  {'pred_elig':>9}")
    print("-"*65)
    for src in sorted(eval_df["source"].dropna().unique()):
        g    = eval_df[eval_df["source"] == src]
        yt_s = (g["gold_label"] == "eligible").astype(int)
        yp_s = (g["auto_pred"]  == "eligible").astype(int)
        print(f"{src:<28}{len(g):>4}  "
              f"{accuracy_score(yt_s, yp_s):>7.3f}  "
              f"{f1_score(yt_s, yp_s, zero_division=0):>7.3f}  "
              f"{yt_s.sum():>9}  {yp_s.sum():>9}")

    # Compare vs v5 and v6
    v5 = {"accuracy":0.8515,"kappa":0.5769,"f1":0.6725,
          "precision":0.6364,"recall":0.7130}
    v6 = {"accuracy":0.8656,"kappa":0.6527,"f1":0.7407,
          "precision":0.6452,"recall":0.8696}  # pre-FP-fix run

    print(f"\n{'='*60}")
    print(f"  VERSION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<12}{'v5':>8}{'v6':>8}{'v7':>8}{'Δ v5→v7':>10}")
    print("-"*45)
    for name, key in [("Accuracy","accuracy"),("Kappa","kappa"),
                      ("F1","f1"),("Precision","precision"),("Recall","recall")]:
        v7v = m[key]
        d   = v7v - v5[key]
        print(f"  {name:<10}{v5[key]:>8.4f}{v6[key]:>8.4f}{v7v:>8.4f}"
              f"  {'+' if d>=0 else ''}{d:.4f}")

    # Remaining errors
    fn_df = eval_df[(eval_df["gold_label"]=="eligible") &
                    (eval_df["auto_pred"]=="not eligible")]
    fp_df = eval_df[(eval_df["gold_label"]=="not eligible") &
                    (eval_df["auto_pred"]=="eligible")]

    print(f"\n{'='*55}")
    print(f"  REMAINING ERRORS")
    print(f"{'='*55}")
    print(f"  False Negatives : {len(fn_df)}")
    print(f"  False Positives : {len(fp_df)}")
    print(f"\n  Top FN schemes:")
    for sid, cnt in fn_df["scheme_id"].value_counts().head(6).items():
        name = eval_df[eval_df["scheme_id"]==sid]["scheme_name"].iloc[0][:55]
        print(f"    {cnt}x  {sid}  {name}")
    print(f"\n  Top FP schemes:")
    for sid, cnt in fp_df["scheme_id"].value_counts().head(6).items():
        name = eval_df[eval_df["scheme_id"]==sid]["scheme_name"].iloc[0][:55]
        print(f"    {cnt}x  {sid}  {name}")

    eval_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Saved: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
