import os
import json
import uuid
from datetime import datetime, timezone
from config import Config

class FineService:
    """
    Centralized Fine (Challan) Calculation and Generation Service.
    Automatically generates legal statutory fines for confirmed traffic violations
    under the Motor Vehicles Act, 1988 (as amended in 2019) and configured jurisdictions.
    """

    def __init__(self, schedule_path=None, default_jurisdiction=None):
        self.schedule_path = schedule_path or getattr(Config, 'FINE_SCHEDULE_PATH', os.path.join(Config.BASE_DIR, 'configs', 'fine_schedule.json'))
        self.default_jurisdiction = default_jurisdiction or getattr(Config, 'DEFAULT_JURISDICTION', 'India (MV Act 2019)')
        self.schedule = {}
        self.load_schedule()

    def load_schedule(self):
        """Loads or reloads the centralized fine schedule from JSON file."""
        if os.path.exists(self.schedule_path):
            try:
                with open(self.schedule_path, 'r', encoding='utf-8') as f:
                    self.schedule = json.load(f)
                return True
            except Exception as e:
                print(f"[FineService] Warning: Failed to load fine schedule from {self.schedule_path}: {e}")
        
        # Fallback to verified statutory schedule under MV Act 2019
        self.schedule = {
            "version": "2019.1",
            "default_jurisdiction": self.default_jurisdiction,
            "rules": {
                "no_helmet": {
                    "violation_type": "no_helmet",
                    "display_name": "Riding Without Protective Headgear (No Helmet)",
                    "jurisdiction": "India (MV Act 2019)",
                    "fine_amount": 1000.0,
                    "currency": "INR",
                    "legal_provision": "Section 194D, Motor Vehicles Act, 1988 (as amended by Act 32 of 2019)",
                    "effective_date": "2019-09-01",
                    "official_source": "The Motor Vehicles (Amendment) Act, 2019 (Act No. 32 of 2019), MoRTH",
                    "notes": "Penalty of ₹1,000 and disqualification of driver license for 3 months.",
                    "active": True
                },
                "triple_riding": {
                    "violation_type": "triple_riding",
                    "display_name": "Carrying More Than Permitted Riders (Triple Riding)",
                    "jurisdiction": "India (MV Act 2019)",
                    "fine_amount": 1000.0,
                    "currency": "INR",
                    "legal_provision": "Section 194C, Motor Vehicles Act, 1988 (as amended by Act 32 of 2019)",
                    "effective_date": "2019-09-01",
                    "official_source": "The Motor Vehicles (Amendment) Act, 2019 (Act No. 32 of 2019), MoRTH",
                    "notes": "Penalty of ₹1,000 and disqualification of driver license for 3 months.",
                    "active": True
                },
                "wrong_side": {
                    "violation_type": "wrong_side",
                    "display_name": "Driving Against Authorised Flow of Traffic (Wrong Direction)",
                    "jurisdiction": "India (MV Act 2019)",
                    "fine_amount": 5000.0,
                    "currency": "INR",
                    "legal_provision": "Section 184(f) r/w Section 119/177, Motor Vehicles Act, 1988 (as amended by Act 32 of 2019)",
                    "effective_date": "2019-09-01",
                    "official_source": "The Motor Vehicles (Amendment) Act, 2019 (Act No. 32 of 2019), MoRTH",
                    "notes": "Dangerous driving punishable with fine up to ₹5,000 for first offence.",
                    "active": True
                }
            },
            "jurisdiction_overrides": {}
        }
        return False

    def update_rule(self, violation_type, fine_amount=None, legal_provision=None, jurisdiction=None, active=None, official_source=None):
        """Allows authorized administrators to update fine rules dynamically without altering detection code."""
        v_key = self._normalize_violation_type(violation_type)
        if 'rules' not in self.schedule:
            self.schedule['rules'] = {}
        if v_key not in self.schedule['rules']:
            self.schedule['rules'][v_key] = {
                "violation_type": v_key,
                "display_name": v_key.replace('_', ' ').title(),
                "jurisdiction": jurisdiction or self.default_jurisdiction,
                "fine_amount": float(fine_amount or 0.0),
                "currency": "INR",
                "legal_provision": legal_provision or "Motor Vehicles Act, 1988",
                "effective_date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                "official_source": official_source or "Administrative Order",
                "active": active if active is not None else True
            }
        else:
            rule = self.schedule['rules'][v_key]
            if fine_amount is not None:
                rule['fine_amount'] = float(fine_amount)
            if legal_provision is not None:
                rule['legal_provision'] = legal_provision
            if jurisdiction is not None:
                rule['jurisdiction'] = jurisdiction
            if active is not None:
                rule['active'] = bool(active)
            if official_source is not None:
                rule['official_source'] = official_source

        # Persist updated schedule to disk
        try:
            os.makedirs(os.path.dirname(self.schedule_path), exist_ok=True)
            with open(self.schedule_path, 'w', encoding='utf-8') as f:
                json.dump(self.schedule, f, indent=2)
            return True
        except Exception as e:
            print(f"[FineService] Warning: Could not save schedule to {self.schedule_path}: {e}")
            return False

    @staticmethod
    def _normalize_violation_type(vtype):
        """Normalizes various violation type synonyms to canonical identifiers."""
        if not vtype:
            return ""
        vt = str(vtype).lower().strip().replace('-', '_').replace(' ', '_')
        if 'helmet' in vt:
            return 'no_helmet'
        if 'triple' in vt:
            return 'triple_riding'
        if 'wrong' in vt or 'direction' in vt or 'side' in vt:
            return 'wrong_side'
        return vt

    def get_fine_rule(self, violation_type, jurisdiction=None):
        """
        Retrieves the verified fine rule for the given violation type and jurisdiction.
        Returns the rule dict if active, or 'FINE_CONFIGURATION_REQUIRED' if missing/inactive.
        """
        v_key = self._normalize_violation_type(violation_type)
        rules = self.schedule.get('rules', {})
        rule = rules.get(v_key)

        if not rule or not rule.get('active', True):
            return "FINE_CONFIGURATION_REQUIRED"

        # Check for state/jurisdiction overrides
        target_jur = jurisdiction or self.default_jurisdiction
        overrides = self.schedule.get('jurisdiction_overrides', {}).get(target_jur, {}).get(v_key)
        
        final_rule = dict(rule)
        if overrides:
            final_rule.update(overrides)
            final_rule['jurisdiction'] = target_jur

        return final_rule

    def generate_fine_for_violation(self, violation, job=None, db=None, jurisdiction=None):
        """
        Generates a verified fine record for a single confirmed violation.
        
        Returns:
            dict: The created or retrieved fine record, or None / dict with error if ineligible.
        """
        if not violation:
            return None

        # 1. Eligibility Check: Violation must be confirmed
        rev_status = str(violation.get('review_status', '')).lower()
        if rev_status != 'confirmed':
            # Do not generate fines for unconfirmed or ambiguous detections
            return None

        # 2. Strict Wrong-Direction Requirement:
        # Require reliable direction evidence (tracked motion or confirmed lane rule)
        v_type = self._normalize_violation_type(violation.get('violation_type'))
        if v_type == 'wrong_side':
            if violation.get('is_candidate') or rev_status == 'insufficient_evidence':
                return None
            # If notes or status explicitly denote lack of direction evidence
            reason = str(violation.get('reason', '')).lower()
            if 'unavailable' in reason or 'insufficient' in reason:
                return None

        # 3. Retrieve Fine Schedule Rule
        rule = self.get_fine_rule(v_type, jurisdiction=jurisdiction)
        v_id = violation.get('id')
        job_id = violation.get('job_id') or (job.get('id') if job else 'unknown')

        # 4. Check for Existing Fine (Deduplication / Idempotency)
        if db is not None and v_id is not None:
            existing = db.get_fine_by_violation(v_id)
            if existing:
                return existing

        # 5. Plate Number Extraction:
        # If plate OCR was successful, record plate number; otherwise mark UNREADABLE.
        raw_plate = violation.get('plate_text')
        plate_status = violation.get('plate_status', '')
        if raw_plate and str(raw_plate).strip() and plate_status != 'unreadable':
            registration_number = str(raw_plate).strip().upper()
        else:
            registration_number = "UNREADABLE"

        now = datetime.now(timezone.utc)
        viol_timestamp = violation.get('created_at') or now.strftime('%Y-%m-%d %H:%M:%S')
        evidence_path = violation.get('evidence_path') or violation.get('evidence_image')

        if rule == "FINE_CONFIGURATION_REQUIRED" or not isinstance(rule, dict):
            # Report FINE_CONFIGURATION_REQUIRED instead of inventing an amount
            return {
                "status": "FINE_CONFIGURATION_REQUIRED",
                "fine_id": f"UNCONFIGURED-{v_id or 'VIOL'}",
                "violation_id": v_id,
                "job_id": job_id,
                "violation_type": v_type,
                "registration_number": registration_number,
                "fine_amount": None,
                "currency": "INR",
                "legal_provision": "Fine Amount Unavailable (Rule Configuration Required)",
                "jurisdiction": jurisdiction or self.default_jurisdiction,
                "fine_rule_version": self.schedule.get('version', '2019.1'),
                "violation_timestamp": str(viol_timestamp),
                "evidence_path": evidence_path,
                "error": f"No active verified fine rule configured for violation '{v_type}'."
            }

        # 6. Generate Unique Fine ID
        # Format: FINE-<YYYYMMDD>-<HEX8>
        fine_uid = f"FINE-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

        fine_record = {
            "fine_id": fine_uid,
            "violation_id": v_id,
            "job_id": job_id,
            "violation_type": v_type,
            "registration_number": registration_number,
            "plate_status": plate_status or ('readable' if registration_number != 'UNREADABLE' else 'unreadable'),
            "plate_confidence": violation.get('plate_confidence'),
            "raw_plate_text": violation.get('raw_plate_text'),
            "plate_crop_path": violation.get('plate_crop_path'),
            "fine_amount": float(rule['fine_amount']),
            "currency": rule.get('currency', 'INR'),
            "legal_provision": rule['legal_provision'],
            "jurisdiction": rule['jurisdiction'],
            "fine_rule_version": self.schedule.get('version', '2019.1'),
            "violation_timestamp": str(viol_timestamp),
            "evidence_path": evidence_path,
            "status": "ISSUED",
            "notes": rule.get('notes', '')
        }

        # 7. Save to Database (SQLite & MongoDB)
        if db is not None and v_id is not None:
            saved_fine = db.create_fine(fine_record)
            return saved_fine or fine_record

        return fine_record

    def get_job_fine_summary(self, job_id, violations=None, db=None, jurisdiction=None):
        """
        Computes the authoritative fine summary and breakdown for an analyzed job/image:
        - Total confirmed violations count
        - Total fine amount (sum of individual distinct fine entries)
        - Breakdown by violation type (count, subtotal, unit fine, legal provision)
        - Distinct vehicle registration plates
        - Incomplete status flag if any fine amount is unconfigured
        - Individual fine entries
        """
        if db is None:
            from services.database_service import DatabaseService
            db = DatabaseService()

        if violations is None:
            violations = db.get_violations_by_job(job_id)

        job = db.get_job(job_id)
        confirmed_violations = [v for v in violations if v.get('review_status') == 'confirmed']
        confirmed_count = len(confirmed_violations)

        fines = []
        has_unconfigured = False
        unconfigured_types = []

        # Generate or retrieve fine for each confirmed violation
        for v in confirmed_violations:
            fine = self.generate_fine_for_violation(v, job=job, db=db, jurisdiction=jurisdiction)
            if fine and isinstance(fine, dict):
                fines.append(fine)
                if fine.get('fine_amount') is None or fine.get('status') == 'FINE_CONFIGURATION_REQUIRED':
                    has_unconfigured = True
                    unconfigured_types.append(fine.get('violation_type', 'unknown'))

        # Sync with persisted fines in database to guarantee completeness
        persisted_fines = db.get_fines_by_job(job_id)
        persisted_by_viol = {f['violation_id']: f for f in persisted_fines}
        for idx, f in enumerate(fines):
            vid = f.get('violation_id')
            if vid in persisted_by_viol:
                fines[idx] = persisted_by_viol[vid]

        # Only include fines belonging to confirmed violations
        confirmed_vids = {v.get('id') for v in confirmed_violations if v.get('id') is not None}
        if confirmed_vids:
            fines = [f for f in fines if f.get('violation_id') in confirmed_vids or not f.get('violation_id')]
        else:
            fines = []

        # Deduplicate fines by violation_id to strictly prevent double-counting
        seen_vids = set()
        deduped_fines = []
        for f in fines:
            vid = f.get('violation_id')
            if vid is not None:
                if vid in seen_vids:
                    continue
                seen_vids.add(vid)
            deduped_fines.append(f)
        fines = deduped_fines

        # Detect any unconfigured or missing fine amounts
        has_unconfigured = False
        unconfigured_types = []
        for f in fines:
            amt = f.get('fine_amount')
            if amt is None or f.get('status') == 'FINE_CONFIGURATION_REQUIRED':
                has_unconfigured = True
                unconfigured_types.append(f.get('violation_type', 'unknown'))

        # Calculate distinct total from valid issued fines
        known_total = 0.0
        for f in fines:
            amt = f.get('fine_amount')
            if amt is not None and f.get('status') == 'ISSUED':
                known_total += float(amt)

        # Categorical breakdown by violation type
        breakdown_map = {}
        for f in fines:
            vt = f.get('violation_type', 'other')
            amt = f.get('fine_amount')
            if vt not in breakdown_map:
                display_name = {
                    'no_helmet': 'Riding Without Protective Headgear (No Helmet)',
                    'triple_riding': 'Triple Riding (Passenger Overload)',
                    'wrong_side': 'Wrong Direction (Contraflow Driving)'
                }.get(vt, vt.replace('_', ' ').title())

                short_name = {
                    'no_helmet': 'No Helmet',
                    'triple_riding': 'Triple Riding',
                    'wrong_side': 'Wrong Direction'
                }.get(vt, vt.replace('_', ' ').title())

                breakdown_map[vt] = {
                    'violation_type': vt,
                    'display_name': display_name,
                    'short_name': short_name,
                    'count': 0,
                    'subtotal': 0.0,
                    'has_unavailable': False,
                    'unit_amount': float(amt) if amt is not None else None,
                    'legal_provision': f.get('legal_provision', '')
                }
            breakdown_map[vt]['count'] += 1
            if amt is not None and f.get('status') == 'ISSUED':
                breakdown_map[vt]['subtotal'] += float(amt)
            else:
                breakdown_map[vt]['has_unavailable'] = True

        breakdown_list = list(breakdown_map.values())

        # Collect vehicle registration numbers
        plates = []
        for f in fines:
            reg = f.get('registration_number')
            if reg and reg != 'UNREADABLE' and reg not in plates:
                plates.append(reg)
        for v in violations:
            pt = v.get('plate_text')
            if pt and str(pt).strip() and str(pt).strip().upper() not in plates and str(pt).strip().upper() != 'UNREADABLE':
                plates.append(str(pt).strip().upper())

        primary_plate = plates[0] if plates else ('UNREADABLE' if fines or violations else None)

        if confirmed_count == 0:
            total_fine_amount = 0.0
            display_status = "TOTAL FINE: ₹0"
            is_incomplete = False
        elif has_unconfigured:
            total_fine_amount = None
            display_status = "TOTAL FINE: INCOMPLETE (Fine amount unavailable)"
            is_incomplete = True
        else:
            total_fine_amount = round(known_total, 2)
            display_status = f"TOTAL FINE: ₹{int(known_total):,}"
            is_incomplete = False

        return {
            'job_id': job_id,
            'confirmed_violations_count': confirmed_count,
            'fines': fines,
            'total_fines': len(fines),
            'total_fine_amount': total_fine_amount,
            'known_subtotal': round(known_total, 2),
            'currency': 'INR',
            'is_incomplete': is_incomplete,
            'display_status': display_status,
            'breakdown': breakdown_list,
            'registration_numbers': plates,
            'primary_plate': primary_plate
        }

    def process_job_fines(self, job_id, violations=None, db=None, jurisdiction=None):
        """
        Processes and generates fines for all eligible confirmed violations in a job.
        
        Returns:
            tuple: (list_of_fine_records, total_fine_amount_inr)
        """
        summary = self.get_job_fine_summary(job_id, violations=violations, db=db, jurisdiction=jurisdiction)
        return summary['fines'], summary['total_fine_amount']
