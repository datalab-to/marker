from typing import Optional, Dict, Any

import yaml

from marker.logger import get_logger

logger = get_logger()


class RuleEngine:
    def __init__(self, rules_path: Optional[str]):
        self.rules: Dict[str, Any] = {}
        if rules_path:
            try:
                self.rules = self._load_rules(rules_path)
            except Exception as e:
                logger.warning(f"Could not load rules from {rules_path}: {e}")
                pass

    def _load_rules(self, rules_path: str) -> Dict[str, Any]:
        with open(rules_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_rules(self, stage: str, default: Any = None) -> Optional[Any]:
        if not self.rules:
            return default
        return self.rules.get("rules", {}).get(stage, default)
