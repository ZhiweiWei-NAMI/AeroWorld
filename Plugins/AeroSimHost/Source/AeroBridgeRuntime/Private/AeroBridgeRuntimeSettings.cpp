#include "AeroBridgeRuntimeSettings.h"

#include "Misc/Paths.h"

UAeroBridgeRuntimeSettings::UAeroBridgeRuntimeSettings()
{
	LowAltitudeConfigRoot = TEXT("Config/LowAltitude");
	AssetCatalogRelativePath = TEXT("Config/LowAltitude/asset_catalog.json");
	WeatherProfilesRelativePath = TEXT("Config/LowAltitude/weather_render_profiles.json");
	MapsRelativeRoot = TEXT("Config/LowAltitude/Maps");
}

FName UAeroBridgeRuntimeSettings::GetCategoryName() const
{
	return TEXT("Plugins");
}
