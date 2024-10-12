import itertools
import json
import os
import random
import string
from faker import Faker
from tqdm import tqdm

# Ensure cache directory exists
def ensure_cache_dir():
    if not os.path.exists("cache"):
        os.makedirs("cache")

# Generators with locale handling
def generate_credit_card_numbers(n, locale="en_US"):
    fake = Faker(locale)
    credit_cards = []
    for _ in range(n):
        formats = [
            lambda: fake.credit_card_number(),
            lambda: "-".join(
                fake.credit_card_number(card_type=None)[i: i + 4]
                for i in range(0, 16, 4)
            ),
            lambda: " ".join(
                fake.credit_card_number(card_type=None)[i: i + 4]
                for i in range(0, 16, 4)
            ),
            lambda: fake.credit_card_number(card_type="visa"),
            lambda: fake.credit_card_number(card_type="mastercard"),
            lambda: fake.credit_card_number(card_type="amex"),
            lambda: fake.credit_card_number(card_type="discover"),
            lambda: fake.credit_card_number(card_type="jcb"),
            lambda: fake.credit_card_number(card_type="diners"),
            lambda: fake.credit_card_number(card_type="unionpay"),
            lambda: "-".join(
                fake.credit_card_number(card_type="amex")[i: i + 4]
                for i in range(0, 15, 4)
            ),
            lambda: " ".join(
                fake.credit_card_number(card_type="amex")[i: i + 4]
                for i in range(0, 15, 4)
            ),
        ]
        cc_format = random.choice(formats)
        credit_card = cc_format()
        credit_cards.append(credit_card)
    return credit_cards

def generate_diverse_dates(n, locale="en_US"):
    fake = Faker(locale)
    dates = []
    for _ in range(n):
        formats = [
            lambda: fake.date(),
            lambda: fake.date(pattern="%Y-%m-%d"),
            lambda: fake.date(pattern="%m/%d/%Y"),
            lambda: fake.date(pattern="%d %b %Y"),
            lambda: fake.date(pattern="%b %d, %Y"),
            lambda: fake.date(pattern="%A, %B %d, %Y"),
            lambda: fake.date(pattern="%d.%m.%Y"),
            lambda: fake.date(pattern="%Y/%m/%d"),
            lambda: fake.date(pattern="%Y.%m.%d"),
        ]
        date_format = random.choice(formats)
        date = date_format()
        dates.append(date)
    return dates

def generate_diverse_datetimes(n, locale="en_US"):
    fake = Faker(locale)
    datetimes = []
    for _ in range(n):
        formats = [
            lambda: fake.date_time().strftime("%Y-%m-%d %H:%M:%S"),
            lambda: fake.date_time().strftime("%m/%d/%Y %I:%M %p"),
            lambda: fake.date_time().strftime("%d %b %Y %H:%M:%S"),
            lambda: fake.date_time().strftime("%a, %d %b %Y %H:%M:%S %Z"),
            lambda: fake.date_time().isoformat(),
            lambda: fake.date_time().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            lambda: fake.date_time_this_century(before_now=True, after_now=False, tzinfo=fake.timezone())
        ]
        datetime_format = random.choice(formats)
        datetime_str = datetime_format()
        datetimes.append(datetime_str)
    return datetimes

def generate_diverse_times(n, locale="en_US"):
    fake = Faker(locale)
    times = []
    for _ in range(n):
        formats = [
            lambda: fake.time(),
            lambda: fake.time(pattern="%H:%M"),
            lambda: fake.time(pattern="%I:%M %p"),
            lambda: fake.time(pattern="%H:%M:%S"),
            lambda: fake.time(pattern="%I:%M:%S %p"),
            lambda: fake.time(pattern="%H:%M:%S.%f"),
            lambda: fake.time(pattern="%H%M hours"),
        ]
        time_format = random.choice(formats)
        time = time_format()
        times.append(time)
    return times

def generate_diverse_customer_ids(n, locale="en_US"):
    fake = Faker(locale)
    customer_ids = []
    for _ in range(n):
        formats = [
            lambda: f"CID-{fake.random_number(digits=6)}",
            lambda: f"CUST{fake.random_number(digits=8)}",
            lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=7)}",
            lambda: f"{fake.random_uppercase_letter()}-{fake.random_number(digits=6)}-{fake.random_uppercase_letter()}",
            lambda: f"C{fake.random_number(digits=3)}-{fake.random_number(digits=4)}-{fake.random_number(digits=3)}",
            lambda: f"{fake.random_uppercase_letter()}{fake.random_lowercase_letter()}-{fake.random_number(digits=5)}",
            lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=4)}{fake.random_uppercase_letter()}{fake.random_number(digits=3)}",
            lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=6)}-{fake.random_uppercase_letter()}{fake.random_lowercase_letter()}",
            lambda: f"{fake.random_uppercase_letter()}{fake.random_lowercase_letter()}{fake.random_number(digits=5)}-{fake.random_uppercase_letter()}",
            lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=3)}-{fake.random_uppercase_letter()}{fake.random_number(digits=4)}-{fake.random_uppercase_letter()}{fake.random_lowercase_letter()}",
            lambda: f"ECOM-{fake.random_number(digits=6)}",
            lambda: f"ID-{fake.random_number(digits=3)}-{fake.random_uppercase_letter()}",
        ]
        customer_id_format = random.choice(formats)
        customer_id = customer_id_format()
        customer_ids.append(customer_id)
    return customer_ids

def generate_gps_coordinates(n, locale="en_US"):
    fake = Faker(locale)
    coordinates = []
    for _ in range(n):
        formats = [
            lambda: f"{fake.latitude()}, {fake.longitude()}",
            lambda: f"{fake.latitude()} {fake.longitude()}",
            lambda: f"({fake.latitude()}, {fake.longitude()})",
            lambda: f"Latitude: {fake.latitude()}, Longitude: {fake.longitude()}",
            lambda: (
                f"{fake.latitude()} N, {fake.longitude()} E"
                if random.random() < 0.5
                else f"{abs(fake.latitude())} S, {abs(fake.longitude())} W"
            ),
            lambda: f"{fake.latitude()},{fake.longitude()}",
            lambda: f"{int(fake.latitude())}.{fake.msisdn()[:6]}, {int(fake.longitude())}.{fake.msisdn()[:6]}",
        ]
        coordinate_format = random.choice(formats)
        try:
            coordinate = coordinate_format()
        except AttributeError:
            coordinate = f"{fake.latitude()}, {fake.longitude()}"
        coordinates.append(coordinate)
    return coordinates

def generate_first_names(n, locale="en_US"):
    fake = Faker(locale)
    first_names = []
    for _ in range(n):
        first_name = fake.first_name()
        first_names.append(first_name)
    return first_names

def generate_last_names(n, locale="en_US"):
    fake = Faker(locale)
    last_names = []
    for _ in range(n):
        formats = [
            lambda: fake.last_name(),
            lambda: f"{fake.last_name()}-{fake.last_name()}",
        ]
        last_name_format = random.choice(formats)
        last_name = last_name_format()
        last_names.append(last_name)
    return last_names

def generate_full_names(n, locale="en_US"):
    fake = Faker(locale)
    full_names = []
    for _ in range(n):
        formats = [
            lambda: f"{fake.first_name()} {fake.last_name()}",
            lambda: f"{fake.first_name()} {fake.first_name()} {fake.last_name()}",
            lambda: f"{fake.first_name()} {fake.last_name()}-{fake.last_name()}",
            lambda: f"{fake.first_name()} {fake.first_name()[0]}. {fake.last_name()}",
            lambda: f"{fake.prefix()} {fake.first_name()} {fake.last_name()}",
            lambda: f"{fake.first_name()} {fake.first_name()[0]}. {fake.middle_name()[0]}. {fake.last_name()}",
        ]
        full_name_format = random.choice(formats)
        full_name = full_name_format()
        full_names.append(full_name)
    return full_names

def generate_diverse_street_addresses(n, locale="en_US"):
    fake = Faker(locale)
    addresses = []
    
    for _ in range(n):
        # Define possible address formats, including state and state abbreviation when supported
        formats = [
            lambda: fake.street_address(),
            lambda: f"{fake.building_number()} {fake.street_name()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.secondary_address()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, Apt. {fake.building_number()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.city()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.postcode()}, {fake.city()}",
            # Attempt to include state if available
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.city()}{', ' + fake.state() if hasattr(fake, 'state') else ''}",
            # Attempt to include state abbreviation if available
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.postcode()}, {fake.city()}{', ' + fake.state_abbr() if hasattr(fake, 'state_abbr') else ''}",
        ]
        
        # Randomly choose an address format
        address_format = random.choice(formats)
        
        # Try to generate the address
        try:
            address = address_format()
        except AttributeError:
            # General fallback: if some format doesn't work due to missing attributes, omit the missing part
            address = fake.street_address()
        
        addresses.append(address)
    
    return addresses

def generate_employee_ids(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: f"EMP{fake.random_number(digits=6)}",
        lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=7)}",
        lambda: f"{fake.random_uppercase_letter()}-{fake.random_number(digits=6)}-{fake.random_uppercase_letter()}",
        lambda: f"{fake.random_uppercase_letter()}{fake.random_lowercase_letter()}-{fake.random_number(digits=5)}",
    ]
    employee_ids = [
        random.choice(formats)() for _ in range(n)
    ]
    return employee_ids

def generate_passwords(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: fake.password(
            length=random.randint(8, 16),
            special_chars=True,
            digits=True,
            upper_case=True,
            lower_case=True,
        ),
        lambda: fake.password(
            length=random.randint(17, 18),
            special_chars=True,
            digits=True,
            upper_case=True,
            lower_case=True,
        ),
    ]
    passwords = [
        random.choice(formats)() for _ in range(n)
    ]
    return passwords

def generate_diverse_api_keys(n, locale="en_US"):
    api_keys = []
    for _ in range(n):
        key_format = random.choice(
            [
                lambda: f"AKIA{''.join(random.choices(string.ascii_uppercase + string.digits, k=16))}",  # AWS Access Key
                lambda: f"AIza{''.join(random.choices(string.ascii_letters + string.digits + '_-', k=37))}",  # Google API Key
                lambda: "".join(
                    random.choices(string.hexdigits, k=32)
                ),  # Azure API Key
                lambda: f"xoxb-{''.join(random.choices(string.digits, k=12))}-{''.join(random.choices(string.digits, k=12))}-{''.join(random.choices(string.ascii_letters + string.digits, k=24))}",  # Slack API Token
                lambda: f"ghp_{''.join(random.choices(string.ascii_letters + string.digits, k=36))}",  # GitHub Personal Access Token
                lambda: f"sk_live_{''.join(random.choices(string.ascii_letters + string.digits, k=24))}",  # Stripe API Key
                lambda: f"SK{''.join(random.choices(string.ascii_letters + string.digits, k=32))}",  # Twilio API Key
                lambda: f"SG.{''.join(random.choices(string.ascii_letters + string.digits + '-_', k=22))}.{''.join(random.choices(string.ascii_letters + string.digits + '-_', k=43))}",  # SendGrid API Key
                lambda: "".join(
                    random.choices(string.hexdigits, k=36)
                ),  # Heroku API Key
                lambda: f"P{''.join(random.choices(string.ascii_uppercase + string.digits, k=7))}",  # PagerDuty API Key
                lambda: f"sq0atp-{''.join(random.choices(string.ascii_uppercase + string.digits + '_-', k=22))}",  # Square API Key
                lambda: f"sl.{''.join(random.choices(string.ascii_letters + string.digits + '-_', k=28))}",  # Dropbox API Key
                lambda: f"fb-{''.join(random.choices(string.ascii_letters + string.digits, k=40))}",  # Facebook API
                lambda: f"lnkd-{''.join(random.choices(string.ascii_letters + string.digits, k=32))}",  # LinkedIn API
                lambda: "".join(
                    random.choices(string.ascii_letters + string.digits + "-_", k=64)
                ),  # DigitalOcean API Key
            ]
        )
        api_key = key_format()
        api_keys.append(api_key)
    return api_keys

def generate_usernames(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: fake.user_name(),
        lambda: fake.first_name().lower() + str(fake.random_number(digits=2)),
        lambda: fake.first_name().lower() + fake.last_name().lower(),
        lambda: fake.first_name().lower() + "." + fake.last_name().lower(),
        lambda: fake.first_name().lower() + "_" + str(fake.random_number(digits=3)),
        lambda: fake.last_name().lower() + str(fake.random_number(digits=2)),
        lambda: fake.last_name().lower() + str(fake.random_number(digits=4)),
        lambda: f"{fake.first_name().lower()}_{fake.random_number(digits=4)}",
        lambda: f"tw_{fake.first_name().lower()}{fake.random_number(digits=3)}",

    ]
    usernames = [
        random.choice(formats)() for _ in range(n)
    ]
    return usernames

def generate_drivers_license_numbers(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: fake.bothify(text="?########", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        lambda: fake.bothify(text="??########", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        lambda: fake.bothify(
            text="?###-####-###-#", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
        lambda: fake.bothify(text="##-######-##", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        lambda: fake.bothify(
            text="?##-####-###-##", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
    ]
    drivers_license_numbers = [
        random.choice(formats)()
        for _ in range(n)
    ]
    return drivers_license_numbers

def generate_ipv4_addresses(n, locale="en_US"):
    fake = Faker(locale)
    ip_addresses = [fake.ipv4() for _ in range(n)]
    return ip_addresses

def generate_ipv6_addresses(n, locale="en_US"):
    fake = Faker(locale)
    ip_addresses = [fake.ipv6() for _ in range(n)]
    return ip_addresses

# Generators from file 2
def generate_account_pins(n, locale="en_US"):
    fake = Faker(locale)
    formats = [lambda: fake.bothify(text="####"), lambda: fake.bothify(text="######")]
    account_pins = [
        random.choice(formats)() for _ in range(n)
    ]
    return account_pins

def generate_cvv_codes(n, locale="en_US"):
    fake = Faker(locale)
    cvv_codes = [
        fake.bothify(text="###") for _ in range(n)
    ]
    return cvv_codes

def generate_bank_routing_numbers(n, locale="en_US"):
    fake = Faker(locale)
    bank_routing_numbers = [
        fake.bothify(text="#########", letters="")
        for _ in range(n)
    ]
    return bank_routing_numbers

def generate_swift_bic_codes(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: fake.bothify(text="????US??", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + fake.bothify(text="###", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        lambda: fake.bothify(text="????GB??", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + fake.bothify(text="###", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        lambda: fake.bothify(text="????DE??", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + fake.bothify(text="###", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ]
    swift_bic_codes = [
        random.choice(formats)() for _ in range(n)
    ]
    return swift_bic_codes

# Generators from file 3
def generate_medical_record_numbers(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: f"MRN-{fake.random_number(digits=6)}",
        lambda: f"MED{fake.random_number(digits=8)}",
        lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=7)}",
    ]
    medical_record_numbers = [
        random.choice(formats)()
        for _ in range(n)
    ]
    return medical_record_numbers

def generate_health_plan_beneficiary_numbers(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: f"HPBN-{fake.random_number(digits=8)}",
        lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=9)}",
    ]
    health_plan_beneficiary_numbers = [
        random.choice(formats)()
        for _ in range(n)
    ]
    return health_plan_beneficiary_numbers

def generate_account_numbers(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: f"ACCT-{fake.random_number(digits=10)}",
        lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=11)}",
    ]
    account_numbers = [
        random.choice(formats)() for _ in range(n)
    ]
    return account_numbers

def generate_certificate_license_numbers(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: f"CERT-{fake.random_number(digits=8)}",
        lambda: f"LIC-{fake.random_uppercase_letter()}{fake.random_number(digits=7)}",
    ]
    certificate_license_numbers = [
        random.choice(formats)()
        for _ in range(n)
    ]
    return certificate_license_numbers

def generate_biometric_identifiers(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: f"BIO-{fake.random_number(digits=10)}",
        lambda: f"{fake.random_uppercase_letter()}{fake.random_number(digits=11)}",
    ]
    biometric_identifiers = [
        random.choice(formats)()
        for _ in range(n)
    ]
    return biometric_identifiers

def generate_web_urls(n, locale="en_US"):
    fake = Faker(locale)
    web_urls = [fake.url() for _ in range(n)]
    return web_urls

def generate_geographic_subdivisions(n, locale="en_US"):
    fake = Faker(locale)
    geographic_subdivisions = [
        fake.city() for _ in range(n)
    ]
    return geographic_subdivisions

def generate_postal_codes(n, locale="en_US"):
    fake = Faker(locale)
    postal_codes = [fake.postcode() for _ in range(n)]
    return postal_codes

def generate_countries(n, locale="en_US"):
    fake = Faker(locale)
    countries = [fake.country() for _ in range(n)]
    return countries

def generate_states(n, locale="en_US"):
    fake = Faker(locale)
    states = []

    for _ in range(n):
        # Define possible formats that include state name and abbreviation
        formats = [
            lambda: fake.state(),  # State name
            lambda: fake.state_abbr()  # State abbreviation
        ]

        # Randomly choose one of the formats
        states_format = random.choice(formats)

        # Try to generate the state or state abbreviation, fallback to None if it fails
        try:
            state = states_format()
        except AttributeError:
            state = None  # Fallback to None if the method is not available for the locale
        
        states.append(state)

    return states
    
def generate_device_identifiers(n, locale="en_US"):
    def generate_imei():
        imei = "".join(str(random.randint(0, 9)) for _ in range(15))
        return imei

    def generate_imsi():
        mcc = str(random.randint(200, 799)).zfill(3)
        mnc = str(random.randint(0, 999)).zfill(3)
        msin = "".join(str(random.randint(0, 9)) for _ in range(9))
        imsi = mcc + mnc + msin
        return imsi

    device_identifiers = [
        random.choice([generate_imei(), generate_imsi()])
        for _ in range(n)
    ]
    return device_identifiers

def generate_unique_identifiers(n, locale="en_US"):
    def generate_unique_id(length):
        chars = string.ascii_uppercase + string.digits
        return "".join(random.choice(chars) for _ in range(length))

    formats = [
        lambda: f"UID-{generate_unique_id(8)}",
        lambda: f"ID{generate_unique_id(10)}",
        lambda: f"{generate_unique_id(4)}-{generate_unique_id(4)}-{generate_unique_id(4)}",
        lambda: f"{generate_unique_id(6)}-{generate_unique_id(6)}",
    ]

    unique_identifiers = [
        random.choice(formats)()
        for _ in range(n)
    ]
    return unique_identifiers

def generate_vehicle_identifiers(n, locale="en_US"):
    def generate_vin():
        chars = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
        vin = ""
        for _ in range(17):
            vin += random.choice(chars)
        return vin

    vehicle_identifiers = [
        generate_vin() for _ in range(n)
    ]
    return vehicle_identifiers

# New generators for additional PII
def generate_email_addresses(n, locale="en_US"):
    fake = Faker(locale)
    email_addresses = [fake.email() for _ in range(n)]
    return email_addresses

def generate_phone_numbers(n, locale="en_US"):
    fake = Faker(locale)
    phone_numbers = [fake.phone_number() for _ in range(n)]
    return phone_numbers

def generate_social_security_numbers(n, locale="en_US"):
    fake = Faker(locale)
    ssn_format = lambda: fake.ssn()
    social_security_numbers = [
        ssn_format() for _ in range(n)
    ]
    return social_security_numbers

def generate_tax_identification_numbers(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: f"{fake.random_number(digits=3)}-{fake.random_number(digits=2)}-{fake.random_number(digits=4)}",  # US TIN
        lambda: f"{fake.random_number(digits=9)}",  # UK NINO
        lambda: f"{fake.random_number(digits=11)}",  # DE Steuer-ID
    ]
    tax_identification_numbers = [
        random.choice(formats)() for _ in range(n)
    ]
    return tax_identification_numbers

def generate_national_identification_numbers(n, locale="en_US"):
    fake = Faker(locale)
    formats = [
        lambda: f"{fake.random_number(digits=9)}",  # General NIN
        lambda: f"{fake.random_number(digits=3)}-{fake.random_number(digits=2)}-{fake.random_number(digits=4)}",  # Alternate NIN
    ]
    national_identification_numbers = [
        random.choice(formats)() for _ in range(n)
    ]
    return national_identification_numbers

def generate_company_names(n, locale="en_US"):
    fake = Faker(locale)
    company_names = [fake.company() for _ in range(n)]
    return company_names

def generate_dates_of_birth(n, locale="en_US"):
    fake = Faker(locale)
    dates_of_birth = [fake.date_of_birth().strftime("%Y-%m-%d") for _ in range(n)]
    return dates_of_birth

def generate_full_addresses(n, locale="en_US"):
    fake = Faker(locale)
    addresses = []
    for _ in range(n):
        # Define a wide variety of address formats
        formats = [
            lambda: f"{fake.street_address()}, {fake.city()}, {fake.state()} {fake.postcode()}, {fake.country()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.city()}, {fake.state_abbr()} {fake.postcode()}",
            lambda: f"{fake.street_address()}, {fake.city()}, {fake.country()}",
            lambda: f"{fake.secondary_address()}, {fake.street_name()}, {fake.city()}, {fake.state()} {fake.postcode()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.city()}, {fake.country()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.city()}, {fake.state_abbr()} {fake.country()}",
            lambda: f"{fake.street_address()}, {fake.city()}, {fake.state()}",
            lambda: f"{fake.street_name()}, {fake.city()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.city()}, {fake.state_abbr()}",
            lambda: f"{fake.secondary_address()}, {fake.street_name()}, {fake.city()}",
            lambda: f"{fake.street_address()}, {fake.city()}",
            lambda: f"{fake.building_number()} {fake.street_name()}, {fake.city()} {fake.postcode()}, {fake.country()}",
            lambda: f"{fake.street_address()}, {fake.state()} {fake.country()}"
        ]

        # Randomly choose an address format
        address_format = random.choice(formats)

        # Try to generate the address and fallback to street address in case of an error
        try:
            address = address_format()
        except AttributeError:
            address = fake.street_address()  # Fallback to basic street address if any part fails
        
        addresses.append(address)
    
    return addresses

class PIIGenerator:
    def __init__(self, locales=None):
        ensure_cache_dir()
        self.locales = locales if locales else ["en_US"]  # Default to en_US if no locales are provided
        self.fake_generators = [Faker(locale) for locale in self.locales]
        self.pii_types = {
            "credit_card_number": generate_credit_card_numbers,
            "date": generate_diverse_dates,
            "date_time": generate_diverse_datetimes,
            "time": generate_diverse_times,
            "customer_id": generate_diverse_customer_ids,
            "coordinate": generate_gps_coordinates,
            "first_name": generate_first_names,
            "last_name": generate_last_names,
            "name": generate_full_names,
            "street_address": generate_diverse_street_addresses,
            "employee_id": generate_employee_ids,
            "password": generate_passwords,
            "api_key": generate_diverse_api_keys,
            "user_name": generate_usernames,
            "license_plate": generate_drivers_license_numbers,
            "ipv4": generate_ipv4_addresses,
            "ipv6": generate_ipv6_addresses,
            "pin": generate_account_pins,
            "cvv": generate_cvv_codes,
            "bank_routing_number": generate_bank_routing_numbers,
            "swift_bic": generate_swift_bic_codes,
            "medical_record_number": generate_medical_record_numbers,
            "health_plan_beneficiary_number": generate_health_plan_beneficiary_numbers,
            "account_number": generate_account_numbers,
            "certificate_license_number": generate_certificate_license_numbers,
            "biometric_identifier": generate_biometric_identifiers,
            "url": generate_web_urls,
            "city": generate_geographic_subdivisions,
            "postcode": generate_postal_codes,
            "device_identifier": generate_device_identifiers,
            "unique_identifier": generate_unique_identifiers,
            "vehicle_identifier": generate_vehicle_identifiers,
            "email": generate_email_addresses,
            "phone_number": generate_phone_numbers,
            "ssn": generate_social_security_numbers,
            "tax_id": generate_tax_identification_numbers,
            "national_id": generate_national_identification_numbers,
            "company_name": generate_company_names,
            "date_of_birth": generate_dates_of_birth,
            "address": generate_full_addresses,
            "country": generate_countries,
            "state": generate_states,
        }

    def add_custom_list(self, name, custom_list):
        self.pii_types[name] = (itertools.cycle, (custom_list,), "list")

    def get_pii_generator(self, name, count=1):
        if name in self.pii_types:
            func = self.pii_types[name]
            for _ in range(count):
                yield func(1, random.choice(self.locales))
        else:
            raise ValueError(f"PII type '{name}' not defined.")

    def sample(self, name, sample_size=1):
        if name not in self.pii_types:
            raise ValueError(f"PII type '{name}' not defined.")
        func = self.pii_types[name]
        return func(sample_size, random.choice(self.locales))

    def get_all_pii_generators(self):
        return [name for name in self.pii_types]

    def print_examples(self):
        examples = {"locales": self.locales, "pii_examples": {}}
        for name, _ in self.pii_types.items():
            examples["pii_examples"][name] = list(self.sample(name, sample_size=2))
        print(json.dumps(examples, indent=2))
