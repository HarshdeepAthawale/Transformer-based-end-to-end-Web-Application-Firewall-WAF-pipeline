"""
GeoIP Lookup Service
"""
from typing import Dict, Optional
from loguru import logger
import os


class GeoIPLookupService:
    """Service for GeoIP lookups"""
    
    def __init__(self):
        self.geoip_db_path = os.getenv("GEOIP_DB_PATH", "./data/GeoLite2-City.mmdb")
        self.geoip_db = None
        self._load_database()
    
    def _load_database(self):
        """Load GeoIP database"""
        try:
            import geoip2.database
            if os.path.exists(self.geoip_db_path):
                self.geoip_db = geoip2.database.Reader(self.geoip_db_path)
                logger.info(f"GeoIP database loaded from {self.geoip_db_path}")
            else:
                logger.warning(f"GeoIP database not found at {self.geoip_db_path}. Geo-fencing will use fallback.")
        except ImportError:
            logger.warning("geoip2 library not installed. Install with: pip install geoip2")
        except Exception as e:
            logger.error(f"Failed to load GeoIP database: {e}")
    
    def lookup(self, ip: str) -> Optional[Dict]:
        """
        Lookup IP address
        Returns: {
            'country_code': str,
            'country_name': str,
            'city': str,
            'latitude': float,
            'longitude': float,
            'asn': str,
            'isp': str
        }
        """
        if not self.geoip_db:
            # Fallback: return None (will be handled by caller)
            return None
        
        try:
            response = self.geoip_db.city(ip)
            
            return {
                'country_code': response.country.iso_code,
                'country_name': response.country.name,
                'city': response.city.name if response.city.name else None,
                'latitude': response.location.latitude,
                'longitude': response.location.longitude,
                'asn': None,  # Would need ASN database
                'isp': None  # Would need ISP database
            }
        except Exception as e:
            logger.debug(f"GeoIP lookup failed for {ip}: {e}")
            return None
    
    def get_country_code(self, ip: str) -> Optional[str]:
        """Get country code for IP"""
        geo_data = self.lookup(ip)
        return geo_data.get('country_code') if geo_data else None
