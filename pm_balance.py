# pm_balance.py
import logging
from decimal import Decimal
from typing import Optional

from web3 import Web3

# Polygon USDC (native on Polygon PoS)
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

USDC_MIN_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    }
]

_w3: Optional[Web3] = None
_usdc_contract = None


def init_web3_once(polygon_rpc_url: str) -> None:
    global _w3, _usdc_contract
    if _w3 is not None and _usdc_contract is not None:
        return

    if not polygon_rpc_url:
        logging.warning("[PM_BAL] missing POLYGON_RPC_URL, skip chain balance check")
        return

    try:
        w3 = Web3(Web3.HTTPProvider(polygon_rpc_url))
        if not w3.is_connected():
            logging.warning("[PM_BAL] web3 not connected, skip chain balance check")
            return

        usdc = w3.eth.contract(address=w3.to_checksum_address(USDC_ADDRESS), abi=USDC_MIN_ABI)
        _w3 = w3
        _usdc_contract = usdc
        logging.info("[PM_BAL] Web3/USDC initialized")
    except Exception as e:
        logging.error(f"[PM_BAL] init web3/usdc failed: {e}")
        _w3 = None
        _usdc_contract = None


def get_wallet_usdc_balance(polygon_rpc_url: str, wallet_address: str) -> Optional[Decimal]:
    """
    Read on-chain USDC wallet balance (USDC units, 6 decimals).
    This is TOTAL wallet balance (doesn't account for locks/allowances/orders).
    """
    if not wallet_address:
        logging.warning("[PM_BAL] missing wallet_address, skip chain balance check")
        return None

    init_web3_once(polygon_rpc_url)
    if _w3 is None or _usdc_contract is None:
        return None

    try:
        raw = _usdc_contract.functions.balanceOf(_w3.to_checksum_address(wallet_address)).call()
        return Decimal(raw) / Decimal(10**6)
    except Exception as e:
        logging.error(f"[PM_BAL] read USDC balance failed: {e}")
        return None
