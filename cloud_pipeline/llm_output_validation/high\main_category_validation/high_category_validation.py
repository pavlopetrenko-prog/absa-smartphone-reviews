def clean_llm_output_batch(raw_output: str) -> Dict[int, str]:
    MAIN_CATEGORIES = {"INFRASTRUCTURE", "PHONE", "GENERAL_FEEDBACK"}
    result = {}

    try:
        if not raw_output or not isinstance(raw_output, str):
            return {}

        # --- basic cleaning ---
        text = raw_output.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
        text = re.sub(r"(?m)(?<=\{|,)\s*(\d+)\s*:", r'"\1":', text)  # fix unquoted keys

        # --- extract inner JSON if model wrapped it with text ---
        if not text.startswith(("{", "[")):
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                text = match.group(1)

        # --- try parsing ---
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)

        # --- Case 1: dict form {"0":"PHONE"} ---
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if (
                    str(k).isdigit()
                    and isinstance(v, str)
                    and v.strip().upper() in MAIN_CATEGORIES
                ):
                    result[int(k)] = v.strip().upper()

        # --- Case 2: list form [{"...": ...}] ---
        elif isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue

                # detect ID-like field (integer or digit string)
                id_val = None
                for val in item.values():
                    if isinstance(val, (int, str)) and str(val).isdigit():
                        id_val = int(val)
                        break
                if id_val is None:
                    id_val = i  # fallback to index

                # detect category-like field (value from MAIN_CATEGORIES)
                cat_val = None
                for val in item.values():
                    if isinstance(val, str) and val.strip().upper() in MAIN_CATEGORIES:
                        cat_val = val.strip().upper()
                        break

                if cat_val:
                    result[id_val] = cat_val

        return result

    except Exception as e:
        print(f"⚠️ clean_llm_output_batch general failure: {e}")
        return {}
