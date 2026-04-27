import json
import logging
import threading
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ─── MOCK DATABASE / STATE ──────────────────────────
# Bankacılık simülasyonu için durum (state) koruyan InMemory Veritabanı
class InMemoryBankDB:
    def __init__(self):
        self._lock = threading.RLock()
        self.accounts = {
            "TR1234567890": {"balance": 15000.0, "owner": "Demo Kullanıcı 1"},
            "TR0987654321": {"balance": 250.0,   "owner": "Demo Kullanıcı 2"},
        }
        self.cards = {
            "CARD_1": {"limit": 5000.0,  "status": "ACTIVE"},
            "CARD_2": {"limit": 10000.0, "status": "FROZEN"},
        }
        self.bills = {
            "SUB_123": {"type": "Elektrik", "amount": 350.0, "status": "UNPAID"},
            "SUB_456": {"type": "Su",       "amount": 120.0, "status": "PAID"},
        }


db = InMemoryBankDB()


# ─── TOOL SCHEMA VE FONKSIYONLARI ───────────────────

class TransferMoneyInput(BaseModel):
    from_account: str = Field(..., description="Gönderici IBAN (Örn: TR1234...)")
    to_account: str = Field(..., description="Alıcı IBAN (Örn: TR9876...)")
    amount: float = Field(..., description="Gönderilecek miktar, pozitif sayı olmalı", gt=0)


def transfer_money(from_account: str, to_account: str, amount: float) -> str:
    """Belirtilen IBAN'lar arasında para transferi yapar."""
    with db._lock:
        if from_account not in db.accounts:
            return f"Hata: Gönderici hesap bulunamadı ({from_account})."

        sender_bal = db.accounts[from_account]["balance"]

        if sender_bal < amount:
            return f"Hata: Yetersiz bakiye. Mevcut bakiye: {sender_bal} TL"

        db.accounts[from_account]["balance"] -= amount

        if to_account in db.accounts:
            db.accounts[to_account]["balance"] += amount
            return (f"Başarılı: {amount} TL tutarında transfer gerçekleşti. "
                    f"{from_account} güncel bakiye: {db.accounts[from_account]['balance']} TL.")
        # Mock dış hesap
        return (f"Başarılı: {amount} TL tutarındaki FAST/EFT işlemi farklı bankaya iletildi. "
                f"Kalan bakiye: {db.accounts[from_account]['balance']} TL.")


class QueryBalanceInput(BaseModel):
    account: str = Field(..., description="Bakiye sorgulanacak IBAN numarası")


def query_balance(account: str) -> str:
    """Mevcut hesap bakiyesini sorgular."""
    with db._lock:
        if account in db.accounts:
            acc = db.accounts[account]
            return f"Hesap Bilgisi ({account}): Bakiye {acc['balance']} TL. Sahibi: {acc['owner']}"
        return f"Hata: Hesap bulunamadı ({account})."


class PayBillInput(BaseModel):
    subscriber_id: str = Field(..., description="Abone/Fatura numarası")
    from_account: str = Field(..., description="Ödemenin çekileceği hesap IBAN'ı")


def pay_bill(subscriber_id: str, from_account: str) -> str:
    """Belirtilen abone numarası için fatura ödemesi yapar."""
    with db._lock:
        if subscriber_id not in db.bills:
            return "Hata: Abone numarasına ait fatura bulunamadı."

        bill = db.bills[subscriber_id]
        if bill["status"] == "PAID":
            return "Bilgi: Bu fatura zaten ödenmiş."

        amount = bill["amount"]

        if from_account not in db.accounts:
            return "Hata: Ödeme yapılacak hesap bulunamadı."

        if db.accounts[from_account]["balance"] < amount:
            return f"Hata: Yetersiz bakiye. Fatura tutarı: {amount} TL"

        db.accounts[from_account]["balance"] -= amount
        bill["status"] = "PAID"

        return (f"Başarılı: {amount} TL tutarındaki {bill['type']} faturanız ödendi. "
                f"Kalan bakiye: {db.accounts[from_account]['balance']} TL")


# ─── TOOL DISPATCHER (ARAÇ YÜRÜTÜCÜ) ────────────────

class ToolDispatcher:
    """LLM'den gelen 'Action' isteklerini çalıştırır."""
    
    def __init__(self):
        self.tools = {
            "transfer_money": {
                "func": transfer_money,
                "schema": TransferMoneyInput,
                "desc": "Başka bir hesaba para gönderir. Parametreler: from_account, to_account, amount."
            },
            "query_balance": {
                "func": query_balance,
                "schema": QueryBalanceInput,
                "desc": "Hesap bakiyesini sorgular. Parametreler: account."
            },
            "pay_bill": {
                "func": pay_bill,
                "schema": PayBillInput,
                "desc": "Fatura ödemesi yapar. Parametreler: subscriber_id, from_account."
            }
        }
        
    def get_tool_descriptions(self) -> str:
        """LLM promptuna enjekte edilecek araç açıklamaları."""
        desc = "KULLANABİLECEĞİN ARAÇLAR (TOOLS):\n"
        for name, data in self.tools.items():
            desc += f"- {name}: {data['desc']}\n"
        return desc
        
    def execute(self, tool_name: str, parameters_json: str) -> str:
        """LLM'in seçtiği aracı validasyon yaparak çalıştırır."""
        if tool_name not in self.tools:
            return f"Sistem Hatası: '{tool_name}' adında bir araç bulunamadı."
            
        tool = self.tools[tool_name]
        
        try:
            # JSON'u parse et
            params_dict = json.loads(parameters_json)
            # Pydantic ile validate et
            validated_params = tool["schema"](**params_dict)
        except json.JSONDecodeError:
            return "Sistem Hatası: Parametreler geçerli bir JSON formatında değil."
        except ValidationError as e:
            # LLM'e Pydantic hata mesajını geri dönüyoruz ki kendini düzeltsin
            return f"Parametre Hatası (Bilgi Eksik): {str(e)}"
            
        try:
            # Fonksiyonu çalıştır
            result = tool["func"](**validated_params.model_dump())
            logger.info(f"Tool Execute: {tool_name} -> {result}")
            return result
        except Exception as e:
            return f"Sistem Hatası: İşlem sırasında bir hata oluştu ({str(e)})."
