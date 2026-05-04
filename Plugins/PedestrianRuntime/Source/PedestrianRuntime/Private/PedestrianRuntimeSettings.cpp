#include "PedestrianRuntimeSettings.h"

#include "CrowdAppearancePool.h"
#include "CrowdRoleProfile.h"
#include "PedestrianCharacter.h"

UPedestrianRuntimeSettings::UPedestrianRuntimeSettings()
{
	DefaultPedestrianClass = TSoftClassPtr<APedestrianCharacter>(
		FSoftObjectPath(TEXT("/AeroWorldContent/Blueprints/Pedestrians/BP_AW_Pedestrian_CityOps_01.BP_AW_Pedestrian_CityOps_01_C")));
	DefaultSpawnVariantId = FName(TEXT("adult_female_commuter"));
	DefaultCrowdAppearancePool = TSoftObjectPtr<UCrowdAppearancePool>(
		FSoftObjectPath(TEXT("/AeroWorldContent/DataAssets/Crowd/DA_AW_CrowdAppearancePool_CityOps_01.DA_AW_CrowdAppearancePool_CityOps_01")));
	DefaultCrowdRoleProfile = TSoftObjectPtr<UCrowdRoleProfile>(
		FSoftObjectPath(TEXT("/AeroWorldContent/DataAssets/Crowd/DA_AW_CrowdRoleProfile_CityOps_Default_01.DA_AW_CrowdRoleProfile_CityOps_Default_01")));
}

FName UPedestrianRuntimeSettings::GetCategoryName() const
{
	return TEXT("Plugins");
}
