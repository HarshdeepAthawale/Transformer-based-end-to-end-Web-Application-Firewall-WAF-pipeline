"""
ISO 3166-1 alpha-2 country code to country name mapping.
Used for geographic stats when GeoRule or GeoIP country_name is not available.
"""
from typing import Dict

# ISO 3166-1 alpha-2 -> country name (common names)
COUNTRY_NAMES: Dict[str, str] = {
    "AD": "Andorra", "AE": "United Arab Emirates", "AF": "Afghanistan",
    "AG": "Antigua and Barbuda", "AI": "Anguilla", "AL": "Albania",
    "AM": "Armenia", "AO": "Angola", "AR": "Argentina", "AS": "American Samoa",
    "AT": "Austria", "AU": "Australia", "AW": "Aruba", "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina", "BB": "Barbados", "BD": "Bangladesh",
    "BE": "Belgium", "BF": "Burkina Faso", "BG": "Bulgaria", "BH": "Bahrain",
    "BI": "Burundi", "BJ": "Benin", "BM": "Bermuda", "BN": "Brunei",
    "BO": "Bolivia", "BQ": "Caribbean Netherlands", "BR": "Brazil",
    "BS": "Bahamas", "BT": "Bhutan", "BW": "Botswana", "BY": "Belarus",
    "BZ": "Belize", "CA": "Canada", "CD": "Democratic Republic of the Congo",
    "CF": "Central African Republic", "CG": "Republic of the Congo",
    "CH": "Switzerland", "CI": "Ivory Coast", "CL": "Chile",
    "CM": "Cameroon", "CN": "China", "CO": "Colombia", "CR": "Costa Rica",
    "CU": "Cuba", "CV": "Cape Verde", "CW": "Curaçao", "CY": "Cyprus",
    "CZ": "Czech Republic", "DE": "Germany", "DJ": "Djibouti",
    "DK": "Denmark", "DM": "Dominica", "DO": "Dominican Republic",
    "DZ": "Algeria", "EC": "Ecuador", "EE": "Estonia", "EG": "Egypt",
    "EH": "Western Sahara", "ER": "Eritrea", "ES": "Spain", "ET": "Ethiopia",
    "FI": "Finland", "FJ": "Fiji", "FK": "Falkland Islands",
    "FM": "Micronesia", "FO": "Faroe Islands", "FR": "France",
    "GA": "Gabon", "GB": "United Kingdom", "GD": "Grenada",
    "GE": "Georgia", "GF": "French Guiana", "GG": "Guernsey",
    "GH": "Ghana", "GI": "Gibraltar", "GL": "Greenland", "GM": "Gambia",
    "GN": "Guinea", "GQ": "Equatorial Guinea", "GR": "Greece",
    "GT": "Guatemala", "GU": "Guam", "GW": "Guinea-Bissau",
    "GY": "Guyana", "HK": "Hong Kong", "HN": "Honduras", "HR": "Croatia",
    "HT": "Haiti", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland",
    "IL": "Israel", "IM": "Isle of Man", "IN": "India", "IQ": "Iraq",
    "IR": "Iran", "IS": "Iceland", "IT": "Italy", "JE": "Jersey",
    "JM": "Jamaica", "JO": "Jordan", "JP": "Japan", "KE": "Kenya",
    "KG": "Kyrgyzstan", "KH": "Cambodia", "KI": "Kiribati", "KM": "Comoros",
    "KN": "Saint Kitts and Nevis", "KP": "North Korea", "KR": "South Korea",
    "KW": "Kuwait", "KY": "Cayman Islands", "KZ": "Kazakhstan",
    "LA": "Laos", "LB": "Lebanon", "LC": "Saint Lucia", "LI": "Liechtenstein",
    "LK": "Sri Lanka", "LR": "Liberia", "LS": "Lesotho", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "LY": "Libya", "MA": "Morocco",
    "MC": "Monaco", "MD": "Moldova", "ME": "Montenegro", "MG": "Madagascar",
    "MH": "Marshall Islands", "MK": "North Macedonia", "ML": "Mali",
    "MM": "Myanmar", "MN": "Mongolia", "MO": "Macau", "MP": "Northern Mariana Islands",
    "MR": "Mauritania", "MT": "Malta", "MU": "Mauritius", "MV": "Maldives",
    "MW": "Malawi", "MX": "Mexico", "MY": "Malaysia", "MZ": "Mozambique",
    "NA": "Namibia", "NC": "New Caledonia", "NE": "Niger", "NG": "Nigeria",
    "NI": "Nicaragua", "NL": "Netherlands", "NO": "Norway", "NP": "Nepal",
    "NR": "Nauru", "NU": "Niue", "NZ": "New Zealand", "OM": "Oman",
    "PA": "Panama", "PE": "Peru", "PF": "French Polynesia", "PG": "Papua New Guinea",
    "PH": "Philippines", "PK": "Pakistan", "PL": "Poland", "PR": "Puerto Rico",
    "PS": "Palestine", "PT": "Portugal", "PW": "Palau", "PY": "Paraguay",
    "QA": "Qatar", "RE": "Réunion", "RO": "Romania", "RS": "Serbia",
    "RU": "Russia", "RW": "Rwanda", "SA": "Saudi Arabia", "SB": "Solomon Islands",
    "SC": "Seychelles", "SD": "Sudan", "SE": "Sweden", "SG": "Singapore",
    "SI": "Slovenia", "SK": "Slovakia", "SL": "Sierra Leone",
    "SM": "San Marino", "SN": "Senegal", "SO": "Somalia", "SR": "Suriname",
    "SS": "South Sudan", "ST": "São Tomé and Príncipe", "SV": "El Salvador",
    "SY": "Syria", "SZ": "Eswatini", "TC": "Turks and Caicos Islands",
    "TD": "Chad", "TG": "Togo", "TH": "Thailand", "TJ": "Tajikistan",
    "TL": "Timor-Leste", "TM": "Turkmenistan", "TN": "Tunisia",
    "TO": "Tonga", "TR": "Turkey", "TT": "Trinidad and Tobago",
    "TV": "Tuvalu", "TW": "Taiwan", "TZ": "Tanzania", "UA": "Ukraine",
    "UG": "Uganda", "US": "United States", "UY": "Uruguay", "UZ": "Uzbekistan",
    "VA": "Vatican City", "VC": "Saint Vincent and the Grenadines",
    "VE": "Venezuela", "VG": "British Virgin Islands", "VI": "U.S. Virgin Islands",
    "VN": "Vietnam", "VU": "Vanuatu", "WF": "Wallis and Futuna", "WS": "Samoa",
    "YE": "Yemen", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
}


def get_country_name(country_code: str) -> str:
    """Return country name for ISO 3166-1 alpha-2 code, or the code itself as fallback."""
    if not country_code or len(country_code) != 2:
        return country_code or "Unknown"
    return COUNTRY_NAMES.get(country_code.upper(), country_code)
