# GeoIP Database Setup

The WAF dashboard uses GeoIP lookup to attribute traffic and threats to countries. This enables the **Geo Rules** page (globe, bar chart, Top Threat Countries) to display geographic attack origins.

## Requirements

- **GeoLite2-City** database from MaxMind (free, requires sign-up)
- Python package: `geoip2>=4.7.0` (already in `requirements.txt`)

## Setup

### 1. Create a MaxMind Account

1. Go to [MaxMind GeoLite2](https://www.maxmind.com/en/geolite2/signup)
2. Create a free account
3. Go to "Download Files" → GeoLite2 → GeoLite2 City (MMDB format)

### 2. Download the Database

**Option A: Manual download**

1. Log in to your MaxMind account
2. Download `GeoLite2-City.mmdb`
3. Place it in the project's `data/` directory:
   ```
   data/GeoLite2-City.mmdb
   ```

**Option B: Using the MaxMind geoipupdate tool**

```bash
# Install geoipupdate (Linux/macOS)
# Ubuntu/Debian: sudo apt install geoipupdate
# macOS: brew install geoipupdate

# Configure with your MaxMind license key
# Edit /etc/GeoIP.conf or ~/.geoipupdate.conf
# Then run:
geoipupdate
# Copy the downloaded file to data/GeoLite2-City.mmdb
```

### 3. Configure Path (Optional)

By default, the backend looks for the database at:

```
./data/GeoLite2-City.mmdb
```

To use a different path, set the environment variable:

```bash
export GEOIP_DB_PATH=/path/to/GeoLite2-City.mmdb
```

### 4. Verify

Start the backend and send some traffic. Traffic logs and threats will have `country_code` populated when GeoIP lookup succeeds.

## Development Without GeoIP

If you don't have the GeoIP database, the Geo Rules page will show "No geographic data" by default. To populate sample data for development:

```bash
python scripts/seed_geo_traffic.py
```

This inserts sample traffic and threat records with `country_code` set for countries such as US, IN, CN, RU, etc. The globe, bar chart, and Top Threat Countries will display this seeded data.

## License Note

GeoLite2 databases are free but require a MaxMind account and license key for automated updates. See [MaxMind's license terms](https://www.maxmind.com/en/geolite2/signup) for details.
