"""
Indian Banking & Finance Glossary for Domain Pre-training.

This module contains comprehensive financial terminology used in Indian
banking, payments, investments, and taxation. Used for domain pre-training
to teach the model financial vocabulary.

Categories:
    - UPI & Digital Payments
    - Banking Terms
    - Investment & MF Terms
    - Tax Terms
    - Regulatory Terms

Author: Ranjit Behera
"""

# UPI & Digital Payments
UPI_GLOSSARY = """
UPI: Unified Payments Interface - A real-time payment system developed by NPCI that enables instant money transfers between bank accounts through mobile phones. Launched in 2016.

VPA: Virtual Payment Address - A unique identifier for UPI transactions in the format username@bankhandle (e.g., name@ybl, user@okaxis). Eliminates need to share bank account details.

IMPS: Immediate Payment Service - An instant 24x7 interbank electronic fund transfer service launched by NPCI in 2010. Maximum limit typically Rs.5 lakhs per transaction.

NEFT: National Electronic Funds Transfer - A nationwide payment system facilitating one-to-one funds transfer. Operates in half-hourly batches. No minimum limit, maximum limit varies by bank.

RTGS: Real Time Gross Settlement - For high-value instant fund transfers. Minimum Rs.2 lakhs, no upper limit. Settlement happens on a real-time gross basis.

IFSC: Indian Financial System Code - An 11-character alphanumeric code that uniquely identifies bank branches in India. First 4 characters represent bank, 5th is 0, last 6 identify branch.

UPI ID: Also called VPA. Unique address for receiving UPI payments. Format: username@bankhandle.

UPI PIN: 4 or 6 digit Personal Identification Number set by user to authorize UPI transactions. Never share with anyone.

QR Code: Quick Response Code used for UPI payments. Contains merchant's VPA/UPI ID for easy scanning and payment.

Collect Request: UPI feature where money is requested from another user. Payee initiates the request, payer approves.

UPI Lite: Offline UPI feature for small value transactions up to Rs.500 without internet connectivity.

UPI AutoPay: Mandate-based recurring payments through UPI for subscriptions, EMIs, etc.

P2P: Peer-to-Peer payment between two individuals via UPI.

P2M: Peer-to-Merchant payment for purchasing goods/services via UPI.

BHIM: Bharat Interface for Money - Government's UPI payment app developed by NPCI.

PhonePe: Popular UPI payment app owned by Walmart. Uses @ybl (Yes Bank Limited) handle.

GPay: Google Pay - Google's UPI payment app in India. Uses @okaxis, @okicici, @oksbi handles.

Paytm: Payment app offering UPI, wallet, and banking services. Uses @paytm handle.

CRED: Members-only app for credit card payments. Offers rewards for bill payments.

Transaction Reference Number: Unique 12-digit number assigned to each UPI transaction for tracking.

UTR: Unique Transaction Reference - 16 or 22 digit number for NEFT/RTGS/IMPS transactions.

Beneficiary: The person or entity receiving the fund transfer.

Remitter: The person or entity sending/initiating the fund transfer.
"""

# Banking Terms
BANKING_GLOSSARY = """
CASA: Current Account Savings Account - Low-cost deposits for banks. CASA ratio indicates financial health.

FD: Fixed Deposit - Term deposit where money is locked for a fixed period at predetermined interest rate.

RD: Recurring Deposit - Monthly savings scheme with fixed deposit at regular intervals.

Savings Account: Basic bank account for individuals with variable interest (usually 2.5-4% p.a.).

Current Account: Account for businesses with no interest but unlimited transactions.

NRE Account: Non-Resident External account for NRIs. Foreign currency converted to INR. Tax-free interest.

NRO Account: Non-Resident Ordinary account for NRIs. For income earned in India. Taxable interest.

FCNR: Foreign Currency Non-Resident account. Deposits held in foreign currency. Tax-free.

CIF: Customer Information File - Unique identification number assigned to each customer by bank.

Account Number: Unique number identifying a specific bank account. Usually 11-16 digits.

MICR: Magnetic Ink Character Recognition - 9-digit code on cheques identifying bank, branch, and account.

Cheque: Written order to bank to pay specific amount from account to named person/entity.

DD: Demand Draft - Prepaid negotiable instrument for fund transfer. Safer than cheques.

KYC: Know Your Customer - Mandatory identity verification process for opening bank accounts.

e-KYC: Electronic KYC using Aadhaar-based authentication for faster verification.

CKYC: Central KYC - Centralized repository of KYC records to avoid repeated submissions.

Passbook: Physical record of all transactions in a bank account.

Statement: Periodic summary of all transactions, available monthly or on-demand.

Overdraft: Facility to withdraw more than available balance, up to approved limit. Interest charged on usage.

NACH: National Automated Clearing House - NPCI system for bulk/repetitive payments like EMIs, salaries.

ECS: Electronic Clearing Service - For bulk payments like dividends, salaries. Being replaced by NACH.

Standing Instruction: Automatic recurring payment instruction to bank for regular transfers.

Nomination: Designating a person to receive account balance in case of account holder's death.

Joint Account: Bank account held by two or more individuals together.

Minor Account: Bank account for individuals below 18 years, operated by guardian.

Dormant Account: Account with no transactions for extended period (usually 24 months).

ATM: Automated Teller Machine - For cash withdrawal, balance inquiry, mini statements.

Debit Card: Card linked to bank account for purchases and ATM withdrawals.

Credit Card: Card with pre-approved credit limit for purchases. Interest charged on unpaid balance.

CVV: Card Verification Value - 3-digit security code on back of card.

PIN: Personal Identification Number - 4-digit code for ATM/card transactions.

EMI: Equated Monthly Installment - Fixed monthly payment for loans, calculated on principal and interest.

APR: Annual Percentage Rate - True cost of borrowing including interest and fees.

PLR: Prime Lending Rate - Base rate used by banks to set loan interest rates.

MCLR: Marginal Cost of Funds based Lending Rate - Current benchmark for setting loan rates.

Repo Rate: Rate at which RBI lends to commercial banks. Affects all interest rates.

Reverse Repo Rate: Rate at which RBI borrows from commercial banks.

CRR: Cash Reserve Ratio - Percentage of deposits banks must keep with RBI.

SLR: Statutory Liquidity Ratio - Percentage of deposits banks must invest in government securities.

NPA: Non-Performing Asset - Loan where interest/principal payment is overdue by 90+ days.

Write-off: When bank removes irrecoverable loan from books. Borrower still liable.

Lien: Bank's right to hold customer's deposits against loan default.

Garnishee Order: Court order to bank to freeze and transfer funds for debt recovery.
"""

# Investment & Mutual Fund Terms
INVESTMENT_GLOSSARY = """
Mutual Fund: Investment vehicle that pools money from multiple investors to invest in securities.

AMC: Asset Management Company - Entity that manages mutual fund schemes.

NAV: Net Asset Value - Per unit price of mutual fund. Calculated as (Assets - Liabilities) / Units.

AUM: Assets Under Management - Total market value of assets managed by AMC.

SIP: Systematic Investment Plan - Regular investment of fixed amount in mutual fund at set intervals.

SWP: Systematic Withdrawal Plan - Regular withdrawal of fixed amount from mutual fund.

STP: Systematic Transfer Plan - Regular transfer from one fund to another.

Lumpsum: One-time investment in mutual fund.

ELSS: Equity Linked Savings Scheme - Tax-saving mutual fund with 3-year lock-in under Section 80C.

Debt Fund: Mutual fund investing in fixed income securities like bonds, debentures.

Equity Fund: Mutual fund investing primarily in stocks/shares.

Hybrid Fund: Mutual fund investing in both equity and debt instruments.

Index Fund: Fund that replicates a market index like Nifty 50, Sensex.

ETF: Exchange Traded Fund - Index fund traded on stock exchange like a stock.

NFO: New Fund Offer - Initial offering of a new mutual fund scheme.

Exit Load: Fee charged when selling mutual fund units before specified period.

Expense Ratio: Annual fee charged by AMC as percentage of AUM.

CAGR: Compound Annual Growth Rate - Annualized rate of return accounting for compounding.

XIRR: Extended Internal Rate of Return - Actual return considering irregular cash flows.

Absolute Return: Simple percentage gain/loss without annualization.

Direct Plan: Mutual fund bought directly from AMC without distributor. Lower expense ratio.

Regular Plan: Mutual fund bought through distributor. Higher expense ratio, includes commission.

Growth Option: Mutual fund option where returns are reinvested. No dividends paid.

IDCW: Income Distribution cum Capital Withdrawal - New name for dividend option in mutual funds.

Bluechip: Large, well-established companies with stable growth and dividends.

Smallcap: Companies with market cap below Rs.5,000 crores. Higher risk, higher potential return.

Midcap: Companies with market cap between Rs.5,000-20,000 crores.

Largecap: Companies with market cap above Rs.20,000 crores. More stable.

Flexi-cap: Fund that invests across market caps without restrictions.

Sectoral Fund: Fund investing in specific sector like banking, IT, pharma.

Thematic Fund: Fund investing based on theme like ESG, infrastructure, consumption.

SEBI: Securities and Exchange Board of India - Regulator for securities market.

AMFI: Association of Mutual Funds in India - Industry body for mutual funds.

Demat Account: Dematerialized account holding securities in electronic form.

DP: Depository Participant - Intermediary between investor and depository (NSDL/CDSL).

NSDL: National Securities Depository Limited - Depository for holding securities.

CDSL: Central Depository Services Limited - Another depository for securities.

Zerodha: Largest discount broker in India. Pioneer of zero brokerage model.

Groww: Popular investment platform for mutual funds and stocks.

Upstox: Discount broker and investment platform.

Kuvera: Commission-free mutual fund investment platform.
"""

# Tax Terms
TAX_GLOSSARY = """
PAN: Permanent Account Number - 10-character alphanumeric tax identification issued by Income Tax Dept.

TAN: Tax Deduction Account Number - 10-digit number for entities deducting TDS.

TDS: Tax Deducted at Source - Tax deducted by payer before making payment.

TCS: Tax Collected at Source - Tax collected by seller from buyer on certain transactions.

Form 26AS: Annual tax statement showing TDS/TCS and taxes paid.

Form 16: TDS certificate from employer showing salary and tax deducted.

Form 16A: TDS certificate for non-salary income like interest, commission.

ITR: Income Tax Return - Annual declaration of income and tax liability.

ITR-1 (Sahaj): For salaried individuals with income up to Rs.50 lakhs.

ITR-2: For individuals with capital gains, multiple house properties.

ITR-3: For individuals with business/professional income.

ITR-4 (Sugam): For presumptive taxation under Section 44AD/44ADA.

Assessment Year (AY): Year in which income is assessed and return filed. AY follows FY.

Financial Year (FY): April to March period in which income is earned.

Due Date: Last date for filing ITR. Usually July 31 for non-audit cases.

Belated Return: ITR filed after due date but before end of assessment year.

Revised Return: Correcting errors in originally filed return.

Section 80C: Deduction up to Rs.1.5 lakhs for specified investments (PPF, ELSS, LIC, etc.).

Section 80D: Deduction for health insurance premium. Rs.25,000 self, Rs.50,000 parents (senior).

Section 80E: Deduction for education loan interest. No upper limit.

Section 80G: Deduction for donations to specified charitable organizations.

Section 80TTA: Deduction up to Rs.10,000 on savings account interest.

Section 80TTB: Deduction up to Rs.50,000 on deposit interest for senior citizens.

HRA: House Rent Allowance - Tax-exempt allowance for rental accommodation.

LTA: Leave Travel Allowance - Tax-exempt allowance for domestic travel.

Standard Deduction: Flat deduction of Rs.50,000 from salary income.

Old Tax Regime: Tax regime with deductions and exemptions but higher rates.

New Tax Regime: Simplified regime with lower rates but no deductions.

Capital Gains: Profit from sale of capital assets like property, shares.

LTCG: Long Term Capital Gains - Gains from assets held beyond specified period. Lower tax rate.

STCG: Short Term Capital Gains - Gains from assets held below specified period. Higher tax rate.

Indexation: Adjusting purchase price for inflation to reduce capital gains tax.

GST: Goods and Services Tax - Indirect tax on supply of goods and services.

GSTIN: GST Identification Number - 15-digit unique ID for GST registered businesses.

CGST: Central GST - GST component going to central government.

SGST: State GST - GST component going to state government.

IGST: Integrated GST - For inter-state supplies. Goes to consuming state.

HSN Code: Harmonized System Nomenclature - Product classification code for GST.

SAC Code: Services Accounting Code - Service classification code for GST.

e-Way Bill: Electronic document for movement of goods above Rs.50,000.

Input Tax Credit (ITC): Credit of GST paid on inputs against output GST liability.

Advance Tax: Paying tax in installments during the financial year if liability exceeds Rs.10,000.

Self-Assessment Tax: Balance tax paid at time of filing return.

Interest u/s 234A/B/C: Interest for late filing, short advance tax, deferment of advance tax.

Scrutiny Assessment: Detailed examination of return by tax officer.

Faceless Assessment: Online assessment without physical meeting with tax officer.
"""

# Regulatory & Other Terms
REGULATORY_GLOSSARY = """
RBI: Reserve Bank of India - Central bank of India. Regulates monetary policy, banks, forex.

NPCI: National Payments Corporation of India - Umbrella organization for retail payments.

SEBI: Securities and Exchange Board of India - Regulator for securities market.

IRDAI: Insurance Regulatory and Development Authority of India - Insurance regulator.

PFRDA: Pension Fund Regulatory and Development Authority - Pension regulator.

AMFI: Association of Mutual Funds in India - Mutual fund industry body.

IBA: Indian Banks' Association - Banking industry body.

FEDAI: Foreign Exchange Dealers' Association of India - Forex market body.

FEMA: Foreign Exchange Management Act - Law governing forex transactions.

PMLA: Prevention of Money Laundering Act - Anti-money laundering law.

PPF: Public Provident Fund - Government savings scheme with tax benefits. 15-year lock-in.

EPF: Employees' Provident Fund - Mandatory retirement savings for salaried employees.

EPFO: Employees' Provident Fund Organisation - Administers EPF.

UAN: Universal Account Number - Unique number for EPF members.

NPS: National Pension System - Voluntary pension scheme for retirement.

PRAN: Permanent Retirement Account Number - Unique NPS account number.

Aadhaar: 12-digit unique identification number issued by UIDAI.

UIDAI: Unique Identification Authority of India - Issues Aadhaar.

OTP: One Time Password - Time-sensitive code for transaction verification.

2FA: Two Factor Authentication - Two-step verification for security.

CIBIL Score: Credit score by TransUnion CIBIL. Range 300-900. Above 750 is good.

Credit Report: Detailed history of credit behavior and repayment.

Fintech: Financial Technology - Technology to deliver financial services.

Neobank: Digital-only bank without physical branches.

NBFC: Non-Banking Financial Company - Financial institution not holding banking license.

P2P Lending: Peer-to-Peer lending platform connecting borrowers and lenders directly.

BNPL: Buy Now Pay Later - Short-term financing for purchases.

Digital Rupee (e₹): CBDC (Central Bank Digital Currency) issued by RBI.

Cryptocurrency: Digital currency like Bitcoin, Ethereum. Not legal tender in India.

Blockchain: Distributed ledger technology underlying cryptocurrencies.

AML: Anti-Money Laundering - Regulations to prevent financial crimes.

CFT: Combating Financing of Terrorism - Regulations against terror financing.

PPI: Prepaid Payment Instrument - Wallets, prepaid cards for payments.

Ombudsman: RBI-appointed officer for resolving customer grievances against banks.

Lok Adalat: People's court for settling bank-related disputes amicably.

SARFAESI Act: Law allowing banks to recover NPAs by selling secured assets.

DRT: Debt Recovery Tribunal - For recovery of bank dues.

IBC: Insolvency and Bankruptcy Code - For corporate insolvency resolution.

NCLT: National Company Law Tribunal - Adjudicates corporate disputes.
"""

# Bank-specific terms
BANK_SPECIFIC = """
HDFC Bank: Housing Development Finance Corporation Bank. Largest private sector bank in India.

ICICI Bank: Industrial Credit and Investment Corporation of India Bank. Second largest private bank.

SBI: State Bank of India. Largest public sector bank in India. Government-owned.

Axis Bank: Third largest private sector bank. Previously called UTI Bank.

Kotak Mahindra Bank: Private sector bank founded by Uday Kotak.

Yes Bank: Private sector bank. Had crisis in 2020, now under reconstruction.

IndusInd Bank: Private sector bank headquartered in Pune.

IDFC First Bank: Merged entity of IDFC Bank and Capital First.

Federal Bank: Kerala-based private sector bank.

RBL Bank: Ratnakar Bank Limited. Private sector bank.

Canara Bank: Public sector bank merged with Syndicate Bank.

Bank of Baroda: Public sector bank merged with Dena Bank and Vijaya Bank.

Punjab National Bank: Public sector bank merged with Oriental Bank and United Bank.

Union Bank of India: Public sector bank merged with Andhra Bank and Corporation Bank.

Indian Bank: Public sector bank merged with Allahabad Bank.

UCO Bank: Public sector bank previously called United Commercial Bank.

IDBI Bank: Development bank now owned by LIC.

Payments Bank: Bank that can accept deposits up to Rs.2 lakhs but cannot lend. Examples: Paytm, Airtel.

Small Finance Bank: Bank for unserved/underserved sections. Examples: AU, Ujjivan, Equitas.

NBFCs: Bajaj Finance, HDFC Ltd, Shriram Finance, L&T Finance, Mahindra Finance.
"""


def get_full_glossary() -> str:
    """Get the complete banking and finance glossary."""
    sections = [
        "# UPI & DIGITAL PAYMENTS",
        UPI_GLOSSARY,
        "\n# BANKING TERMS",
        BANKING_GLOSSARY,
        "\n# INVESTMENT & MUTUAL FUNDS",
        INVESTMENT_GLOSSARY,
        "\n# TAX TERMS",
        TAX_GLOSSARY,
        "\n# REGULATORY & OTHER TERMS",
        REGULATORY_GLOSSARY,
        "\n# BANK-SPECIFIC TERMS",
        BANK_SPECIFIC,
    ]
    return "\n".join(sections)


def save_glossary(output_path: str = "data/corpus/glossary/banking_glossary.txt"):
    """Save the glossary to a text file."""
    from pathlib import Path
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(get_full_glossary())
    
    # Count terms
    full_text = get_full_glossary()
    term_count = full_text.count(':') - full_text.count('://') 
    word_count = len(full_text.split())
    
    print(f"✅ Saved glossary to {path}")
    print(f"   Terms: ~{term_count}")
    print(f"   Words: {word_count:,}")
    
    return path


if __name__ == "__main__":
    save_glossary()
