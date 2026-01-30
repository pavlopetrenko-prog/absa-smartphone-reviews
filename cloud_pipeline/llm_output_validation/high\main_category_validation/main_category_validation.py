def clean_llm_output_batch(raw_output: str, high_category: str) -> Dict[int, str]:
    CATEGORY_SETS = {
        "PHONE": {
            "BATTERY",
            "CAMERA",
            "PERFORMANCE",
            "DISPLAY",
            "CONNECTIVITY",
            "DESIGN",
            "SOFTWARE",
            "STORAGE",
            "AUDIO",
        },
        "INFRASTRUCTURE": {
            "RETURNS & EXCHANGES",
            "REFUNDS & CASHBACK",
            "DELIVERY",
            "ORDER TRACKING & COMMUNICATION",
            "CUSTOMER SUPPORT",
            "POLICY CLARITY & TRUSTWORTHINESS",
        },
        "GENERAL_FEEDBACK": {
            "PRICE & VALUE",
            "BRAND TRUST & REPUTATION",
            "OVERALL SATISFACTION",
            "LOYALTY & REPEAT PURCHASE",
            "MARKETING & PROMOTIONS",
            "COMPETITOR COMPARISON",
            "ETHICS & SOCIAL RESPONSIBILITY",
        },
    }

    MAIN_CATEGORIES = CATEGORY_SETS.get(high_category, set())
    result = {}

    try:
        if not raw_output or not isinstance(raw_output, str):
            return {}

        text = raw_output.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
        text = re.sub(r"(?m)(?<=\{|,)\s*(\d+)\s*:", r'"\1":', text)

        if not text.startswith(("{", "[")):
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)

        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if str(k).isdigit() and isinstance(v, str):
                    cat = v.strip().upper()
                    if cat in MAIN_CATEGORIES:
                        result[int(k)] = cat

        elif isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue

                id_val = None
                for val in item.values():
                    if isinstance(val, (int, str)) and str(val).isdigit():
                        id_val = int(val)
                        break
                if id_val is None:
                    id_val = i

                cat_val = None
                for val in item.values():
                    if isinstance(val, str):
                        cat = val.strip().upper()
                        if cat in MAIN_CATEGORIES:
                            cat_val = cat
                            break

                if cat_val:
                    result[id_val] = cat_val

        return result

    except Exception as e:
        print(f"⚠️ clean_llm_output_batch general failure: {e}")
        return {}
