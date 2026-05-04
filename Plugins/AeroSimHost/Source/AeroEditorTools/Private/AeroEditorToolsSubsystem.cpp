#include "AeroEditorToolsSubsystem.h"

#include "AeroCompositeMeshActorBase.h"
#include "AeroDynamicVisibleActorBase.h"
#include "AeroPedNavSemanticSubsystem.h"
#include "AeroTriggerZoneBase.h"
#include "Animation/AnimMontage.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "AssetToolsModule.h"
#include "IAssetTools.h"
#include "BlueprintEditorLibrary.h"
#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Editor.h"
#include "Engine/Blueprint.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/StaticMesh.h"
#include "CrowdAppearancePool.h"
#include "CrowdRoleProfile.h"
#include "Factories/BlueprintFactory.h"
#include "Factories/DataAssetFactory.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "PedestrianCharacter.h"
#include "PedestrianRuntimeSettings.h"
#include "PedestrianVariantCatalog.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Subsystems/EditorAssetSubsystem.h"
#include "UObject/SoftObjectPath.h"
#include "Materials/MaterialInterface.h"

namespace
{
constexpr TCHAR* AeroBootstrapCallingContext = TEXT("AeroWorldContentBootstrap");

constexpr TCHAR* AeroWorldContentRoot = TEXT("/AeroWorldContent");
constexpr TCHAR* AeroWorldContentBlueprintRoot = TEXT("/AeroWorldContent/Blueprints");
constexpr TCHAR* AeroWorldContentTriggerDir = TEXT("/AeroWorldContent/Blueprints/Triggers");
constexpr TCHAR* AeroWorldContentPedestrianDir = TEXT("/AeroWorldContent/Blueprints/Pedestrians");
constexpr TCHAR* AeroWorldContentPropDir = TEXT("/AeroWorldContent/Blueprints/Props");
constexpr TCHAR* AeroWorldContentVehicleDir = TEXT("/AeroWorldContent/Blueprints/Vehicles");
constexpr TCHAR* AeroWorldContentUavDir = TEXT("/AeroWorldContent/Blueprints/UAV");
constexpr TCHAR* AeroWorldContentDataAssetDir = TEXT("/AeroWorldContent/DataAssets");
constexpr TCHAR* AeroWorldContentPedDataDir = TEXT("/AeroWorldContent/DataAssets/Ped");
constexpr TCHAR* AeroWorldContentCrowdDataDir = TEXT("/AeroWorldContent/DataAssets/Crowd");

constexpr TCHAR* ChargerMeshDir = TEXT("/AeroWorldContent/Meshes/Facilities/Charger");
constexpr TCHAR* LandingPadMeshDir = TEXT("/AeroWorldContent/Meshes/Facilities/LandingPad");
constexpr TCHAR* RadioMeshDir = TEXT("/AeroWorldContent/Meshes/Infrastructure/Radio");
constexpr TCHAR* TrafficControlMeshDir = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl");
constexpr TCHAR* RoadworkMeshDir = TEXT("/AeroWorldContent/Meshes/Props/Roadwork");
constexpr TCHAR* ServiceMeshDir = TEXT("/AeroWorldContent/Meshes/Props/Service");
constexpr TCHAR* MiscMeshDir = TEXT("/AeroWorldContent/Meshes/Props/Misc");
constexpr TCHAR* EmergencyVehicleMeshDir = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency");

constexpr TCHAR* TriggerNoFlyAssetPath = TEXT("/AeroWorldContent/Blueprints/Triggers/BP_AW_Trigger_NoFly_Box_01.BP_AW_Trigger_NoFly_Box_01");
constexpr TCHAR* TriggerConstructionAssetPath = TEXT("/AeroWorldContent/Blueprints/Triggers/BP_AW_Trigger_Hazard_Construction_Box_01.BP_AW_Trigger_Hazard_Construction_Box_01");
constexpr TCHAR* TriggerGenericAssetPath = TEXT("/AeroWorldContent/Blueprints/Triggers/BP_AW_Trigger_Hazard_Generic_Box_01.BP_AW_Trigger_Hazard_Generic_Box_01");
constexpr TCHAR* PedBlueprintAssetPath = TEXT("/AeroWorldContent/Blueprints/Pedestrians/BP_AW_Pedestrian_CityOps_01.BP_AW_Pedestrian_CityOps_01");
constexpr TCHAR* VehiclePoliceAssetPath = TEXT("/AeroWorldContent/Blueprints/Vehicles/BP_AW_Vehicle_Emergency_SUV_01.BP_AW_Vehicle_Emergency_SUV_01");
constexpr TCHAR* VehicleAmbulanceAssetPath = TEXT("/AeroWorldContent/Blueprints/Vehicles/BP_AW_Vehicle_Emergency_Ambulance_01.BP_AW_Vehicle_Emergency_Ambulance_01");
constexpr TCHAR* VehicleServiceAssetPath = TEXT("/AeroWorldContent/Blueprints/Vehicles/BP_AW_Vehicle_Service_Box_01.BP_AW_Vehicle_Service_Box_01");
constexpr TCHAR* UavInspectionAssetPath = TEXT("/AeroWorldContent/Blueprints/UAV/BP_AW_UAV_Inspection_Quad_01.BP_AW_UAV_Inspection_Quad_01");
constexpr TCHAR* PropBackpackAssetPath = TEXT("/AeroWorldContent/Blueprints/Props/BP_AW_Prop_Service_Backpack_01.BP_AW_Prop_Service_Backpack_01");
constexpr TCHAR* PropPhoneAssetPath = TEXT("/AeroWorldContent/Blueprints/Props/BP_AW_Prop_Misc_Phone_01.BP_AW_Prop_Misc_Phone_01");
constexpr TCHAR* PropUmbrellaAssetPath = TEXT("/AeroWorldContent/Blueprints/Props/BP_AW_Prop_Misc_Umbrella_01.BP_AW_Prop_Misc_Umbrella_01");
constexpr TCHAR* PropBarrierAssetPath = TEXT("/AeroWorldContent/Blueprints/Props/BP_AW_Prop_Roadwork_Barrier_01.BP_AW_Prop_Roadwork_Barrier_01");
constexpr TCHAR* PedCatalogAssetPath = TEXT("/AeroWorldContent/DataAssets/Ped/DA_AW_PedVariants_CityOps_01.DA_AW_PedVariants_CityOps_01");
constexpr TCHAR* CrowdAppearancePoolAssetPath = TEXT("/AeroWorldContent/DataAssets/Crowd/DA_AW_CrowdAppearancePool_CityOps_01.DA_AW_CrowdAppearancePool_CityOps_01");
constexpr TCHAR* CrowdRoleProfileAssetPath = TEXT("/AeroWorldContent/DataAssets/Crowd/DA_AW_CrowdRoleProfile_CityOps_Default_01.DA_AW_CrowdRoleProfile_CityOps_Default_01");
constexpr TCHAR* DefaultPedSpawnVariantId = TEXT("adult_female_commuter");

constexpr TCHAR* PedBlueprintParentPath = TEXT("/Game/MixamoAssets/Blueprints/BP_PedestrianCharacter.BP_PedestrianCharacter");
constexpr TCHAR* VehicleServiceParentPath = TEXT("/AirSim/VehicleAdv/SUV/SuvCarPawn.SuvCarPawn");
constexpr TCHAR* UavInspectionParentPath = TEXT("/AirSim/Blueprints/BP_FlyingPawn.BP_FlyingPawn");

constexpr TCHAR* PedObserveMontagePath = TEXT("/Game/MixamoAssets/Animations/AM_Observe.AM_Observe");
constexpr TCHAR* PedStartCrossMontagePath = TEXT("/Game/MixamoAssets/Animations/AM_StartCross.AM_StartCross");

constexpr TCHAR* ChargerMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Facilities/Charger/LG_EV_Charger_EVD200SK_LG_EVCharger.LG_EV_Charger_EVD200SK_LG_EVCharger");
constexpr TCHAR* ChargerMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Facilities/Charger/_Imported/LG_EV_Charger_EVD200SK_LG_EVCharger.LG_EV_Charger_EVD200SK_LG_EVCharger");
constexpr TCHAR* ChargerMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Facilities/Charger/SM_AW_Facility_Charger_01.SM_AW_Facility_Charger_01");

constexpr TCHAR* LandingPadMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Facilities/LandingPad/landing-pad_extracted/source/landingpad_intact/landingpad_intact.landingpad_intact");
constexpr TCHAR* LandingPadMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Facilities/LandingPad/_Imported/landingpad_intact.landingpad_intact");
constexpr TCHAR* LandingPadMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Facilities/LandingPad/SM_AW_Facility_LandingPad_01.SM_AW_Facility_LandingPad_01");

constexpr TCHAR* RadioMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Infrastructure/Radio/transmissiontower.transmissiontower");
constexpr TCHAR* RadioMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/Radio/_Imported/transmissiontower.transmissiontower");
constexpr TCHAR* RadioMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/Radio/SM_AW_Facility_RadioTower_01.SM_AW_Facility_RadioTower_01");

constexpr TCHAR* PoliceSignMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/police_sign_trimsheet.police_sign_trimsheet");
constexpr TCHAR* PoliceSignMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/_Imported/police_sign_trimsheet.police_sign_trimsheet");
constexpr TCHAR* PoliceSignMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/SM_AW_Prop_TrafficControl_PoliceSign_01.SM_AW_Prop_TrafficControl_PoliceSign_01");

constexpr TCHAR* TrafficLightMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/traffic-light_extracted/source/TrafficLight.TrafficLight");
constexpr TCHAR* TrafficLightMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/_Imported/TrafficLight.TrafficLight");
constexpr TCHAR* TrafficLightMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/SM_AW_Prop_TrafficControl_TrafficLight_01.SM_AW_Prop_TrafficControl_TrafficLight_01");

constexpr TCHAR* PoliceTapeMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/Police_lines_1001.Police_lines_1001");
constexpr TCHAR* PoliceTapeMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/_Imported/Police_lines_1001.Police_lines_1001");
constexpr TCHAR* PoliceTapeMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/SM_AW_Prop_Incident_PoliceTape_01.SM_AW_Prop_Incident_PoliceTape_01");

constexpr TCHAR* BarrierMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Props/Roadwork/Barrier.Barrier");
constexpr TCHAR* BarrierMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Roadwork/_Imported/Barrier.Barrier");
constexpr TCHAR* BarrierMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Props/Roadwork/SM_AW_Prop_Roadwork_Barrier_01.SM_AW_Prop_Roadwork_Barrier_01");
constexpr TCHAR* ConstructionFenceAuthorityPath = TEXT("/AeroWorldContent/Meshes/Props/Roadwork/SM_AW_Prop_ConstructionFence_01.SM_AW_Prop_ConstructionFence_01");

constexpr TCHAR* DeliveryBagMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Props/Service/fooddeliverybag_pCube1320.fooddeliverybag_pCube1320");
constexpr TCHAR* DeliveryBagMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Service/_Imported/fooddeliverybag_pCube1320.fooddeliverybag_pCube1320");
constexpr TCHAR* DeliveryBagMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Props/Service/SM_AW_Prop_Service_DeliveryBag_01.SM_AW_Prop_Service_DeliveryBag_01");

constexpr TCHAR* AmbulanceMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/ambulance_car_-_low_poly.ambulance_car_-_low_poly");
constexpr TCHAR* AmbulanceMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/ambulance_car_-_low_poly.ambulance_car_-_low_poly");
constexpr TCHAR* AmbulanceMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/SM_AW_Vehicle_Emergency_Ambulance_01.SM_AW_Vehicle_Emergency_Ambulance_01");

constexpr TCHAR* PoliceBodySourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/Body_Body_0.Body_Body_0");
constexpr TCHAR* PoliceBodyImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/Body_Body_0.Body_Body_0");
constexpr TCHAR* PoliceGlassSourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/Glass_Glass_0.Glass_Glass_0");
constexpr TCHAR* PoliceGlassImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/Glass_Glass_0.Glass_Glass_0");
constexpr TCHAR* PoliceInteriorSourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/Interior_Interior_0.Interior_Interior_0");
constexpr TCHAR* PoliceInteriorImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/Interior_Interior_0.Interior_Interior_0");
constexpr TCHAR* PoliceShadowSourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/Shadow_Shadow_0.Shadow_Shadow_0");
constexpr TCHAR* PoliceShadowImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/Shadow_Shadow_0.Shadow_Shadow_0");
constexpr TCHAR* PoliceWheelFlSourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/Wheel_FL_Texture_0.Wheel_FL_Texture_0");
constexpr TCHAR* PoliceWheelFlImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/Wheel_FL_Texture_0.Wheel_FL_Texture_0");
constexpr TCHAR* PoliceWheelFrSourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/Wheel_FR_Texture_0.Wheel_FR_Texture_0");
constexpr TCHAR* PoliceWheelFrImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/Wheel_FR_Texture_0.Wheel_FR_Texture_0");
constexpr TCHAR* PoliceWheelRlSourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/Wheel_RL_Texture_0.Wheel_RL_Texture_0");
constexpr TCHAR* PoliceWheelRlImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/Wheel_RL_Texture_0.Wheel_RL_Texture_0");
constexpr TCHAR* PoliceWheelRrSourcePath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/Wheel_RR_Texture_0.Wheel_RR_Texture_0");
constexpr TCHAR* PoliceWheelRrImportedPath = TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/_Imported/Wheel_RR_Texture_0.Wheel_RR_Texture_0");

constexpr TCHAR* BackpackPrimarySourcePath = TEXT("/AeroWorldContent/Meshes/Props/Service/LP_Mochila.LP_Mochila");
constexpr TCHAR* BackpackPrimaryImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Service/_Imported/LP_Mochila.LP_Mochila");
constexpr TCHAR* BackpackExtrasSourcePath = TEXT("/AeroWorldContent/Meshes/Props/Service/LP_Extras.LP_Extras");
constexpr TCHAR* BackpackExtrasImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Service/_Imported/LP_Extras.LP_Extras");
constexpr TCHAR* BackpackRespaldoSourcePath = TEXT("/AeroWorldContent/Meshes/Props/Service/LP_Respaldo.LP_Respaldo");
constexpr TCHAR* BackpackRespaldoImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Service/_Imported/LP_Respaldo.LP_Respaldo");
constexpr TCHAR* BackpackRespaldo2SourcePath = TEXT("/AeroWorldContent/Meshes/Props/Service/LP_Respaldo2.LP_Respaldo2");
constexpr TCHAR* BackpackRespaldo2ImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Service/_Imported/LP_Respaldo2.LP_Respaldo2");
constexpr TCHAR* BackpackStrapAdjustSourcePath = TEXT("/AeroWorldContent/Meshes/Props/Service/LP_TirasAjuste.LP_TirasAjuste");
constexpr TCHAR* BackpackStrapAdjustImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Service/_Imported/LP_TirasAjuste.LP_TirasAjuste");
constexpr TCHAR* BackpackStrapBackSourcePath = TEXT("/AeroWorldContent/Meshes/Props/Service/LP_Tiras_Espalda.LP_Tiras_Espalda");
constexpr TCHAR* BackpackStrapBackImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Service/_Imported/LP_Tiras_Espalda.LP_Tiras_Espalda");

constexpr TCHAR* PhonePrimarySourcePath = TEXT("/AeroWorldContent/Meshes/Props/Misc/cartoonphone_Cube.cartoonphone_Cube");
constexpr TCHAR* PhonePrimaryImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Misc/_Imported/cartoonphone_Cube.cartoonphone_Cube");
constexpr TCHAR* PhonePart01SourcePath = TEXT("/AeroWorldContent/Meshes/Props/Misc/cartoonphone_Cube_001.cartoonphone_Cube_001");
constexpr TCHAR* PhonePart01ImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Misc/_Imported/cartoonphone_Cube_001.cartoonphone_Cube_001");
constexpr TCHAR* PhonePart02SourcePath = TEXT("/AeroWorldContent/Meshes/Props/Misc/cartoonphone_Cube_002.cartoonphone_Cube_002");
constexpr TCHAR* PhonePart02ImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Misc/_Imported/cartoonphone_Cube_002.cartoonphone_Cube_002");

constexpr TCHAR* UmbrellaPrimarySourcePath = TEXT("/AeroWorldContent/Meshes/Props/Misc/umbrella_Panel_001.umbrella_Panel_001");
constexpr TCHAR* UmbrellaPrimaryImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Misc/_Imported/umbrella_Panel_001.umbrella_Panel_001");
constexpr TCHAR* UmbrellaPart01SourcePath = TEXT("/AeroWorldContent/Meshes/Props/Misc/umbrella_RibSpring_001.umbrella_RibSpring_001");
constexpr TCHAR* UmbrellaPart01ImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Misc/_Imported/umbrella_RibSpring_001.umbrella_RibSpring_001");
constexpr TCHAR* UmbrellaPart02SourcePath = TEXT("/AeroWorldContent/Meshes/Props/Misc/umbrella_Shaft_001.umbrella_Shaft_001");
constexpr TCHAR* UmbrellaPart02ImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Misc/_Imported/umbrella_Shaft_001.umbrella_Shaft_001");

constexpr TCHAR* TrafficConeMeshSourcePath = TEXT("/AeroWorldContent/Meshes/Props/Roadwork/traffic-cones_extracted/source/Traffic_Cones_Traffic_Cone_2_1_001.Traffic_Cones_Traffic_Cone_2_1_001");
constexpr TCHAR* TrafficConeMeshImportedPath = TEXT("/AeroWorldContent/Meshes/Props/Roadwork/_Imported/Traffic_Cones_Traffic_Cone_2_1_001.Traffic_Cones_Traffic_Cone_2_1_001");
constexpr TCHAR* TrafficConeMeshAuthorityPath = TEXT("/AeroWorldContent/Meshes/Props/Roadwork/SM_AW_Prop_TrafficCone_01.SM_AW_Prop_TrafficCone_01");
constexpr TCHAR* TrafficConeMaterialPath = TEXT("/AeroWorldContent/Meshes/Props/Roadwork/traffic-cones_extracted/textures/Traffic_Cones_1_1_BaseColor_Mat.Traffic_Cones_1_1_BaseColor_Mat");
constexpr TCHAR* TrafficLightMaterialPath = TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/traffic-light_extracted/textures/TrafficLight_D_tga_Mat.TrafficLight_D_tga_Mat");

FString GetMapConfigDir(const FString& MapId)
{
	return FPaths::Combine(FPaths::ProjectDir(), TEXT("Config/LowAltitude/Maps"), MapId);
}

FString GetAssetCatalogPath()
{
	return FPaths::Combine(FPaths::ProjectDir(), TEXT("Config/LowAltitude/asset_catalog.json"));
}

bool LoadJsonObjectFromFile(const FString& FilePath, TSharedPtr<FJsonObject>& OutObject, FString& OutError)
{
	FString Content;
	if (!FFileHelper::LoadFileToString(Content, *FilePath))
	{
		OutError = FString::Printf(TEXT("Failed to read JSON file: %s"), *FilePath);
		return false;
	}

	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Content);
	if (!FJsonSerializer::Deserialize(Reader, OutObject) || !OutObject.IsValid())
	{
		OutError = FString::Printf(TEXT("Failed to parse JSON file: %s"), *FilePath);
		return false;
	}

	return true;
}

UEditorAssetSubsystem* ResolveEditorAssetSubsystem(FString& OutError)
{
	if (GEditor == nullptr)
	{
		OutError = TEXT("GEditor is unavailable.");
		return nullptr;
	}

	UEditorAssetSubsystem* AssetSubsystem = GEditor->GetEditorSubsystem<UEditorAssetSubsystem>();
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
	}

	return AssetSubsystem;
}

bool EnsureContentDirectory(UEditorAssetSubsystem* AssetSubsystem, const FString& DirectoryPath, FString& OutError)
{
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
		return false;
	}

	if (AssetSubsystem->DoesDirectoryExist(DirectoryPath))
	{
		return true;
	}

	if (!AssetSubsystem->MakeDirectory(DirectoryPath))
	{
		OutError = FString::Printf(TEXT("Failed to create content directory '%s'."), *DirectoryPath);
		return false;
	}

	return true;
}

FString GetPackageNameFromObjectPath(const FString& AssetObjectPath)
{
	return FPackageName::ObjectPathToPackageName(AssetObjectPath);
}

FString GetPackagePathFromObjectPath(const FString& AssetObjectPath)
{
	return FPackageName::GetLongPackagePath(GetPackageNameFromObjectPath(AssetObjectPath));
}

FString GetAssetNameFromObjectPath(const FString& AssetObjectPath)
{
	return FPackageName::ObjectPathToObjectName(AssetObjectPath);
}

UObject* LoadAssetChecked(UEditorAssetSubsystem* AssetSubsystem, const FString& AssetPath, const TCHAR* ExpectedTypeName, FString& OutError)
{
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
		return nullptr;
	}

	UObject* Asset = AssetSubsystem->LoadAsset(AssetPath);
	if (Asset == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to load %s asset '%s'."), ExpectedTypeName, *AssetPath);
	}

	return Asset;
}

UClass* LoadParentBlueprintClass(UEditorAssetSubsystem* AssetSubsystem, const FString& ParentAssetPath, FString& OutError)
{
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
		return nullptr;
	}

	UClass* ParentClass = AssetSubsystem->LoadBlueprintClass(ParentAssetPath);
	if (ParentClass == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to load parent blueprint class '%s'."), *ParentAssetPath);
	}

	return ParentClass;
}

bool RenameLoadedAssetToPath(UObject* Asset, const FString& DestinationObjectPath, FString& OutError)
{
	if (Asset == nullptr)
	{
		OutError = TEXT("Cannot rename null asset.");
		return false;
	}

	TArray<FAssetRenameData> RenameData;
	RenameData.Emplace(Asset, GetPackagePathFromObjectPath(DestinationObjectPath), GetAssetNameFromObjectPath(DestinationObjectPath));
	return FAssetToolsModule::GetModule().Get().RenameAssets(RenameData);
}

UObject* EnsureImportedAssetMoved(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& SourceObjectPath,
	const FString& ImportedObjectPath,
	FString& OutError)
{
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
		return nullptr;
	}

	if (AssetSubsystem->DoesAssetExist(ImportedObjectPath))
	{
		return AssetSubsystem->LoadAsset(ImportedObjectPath);
	}

	if (!AssetSubsystem->DoesAssetExist(SourceObjectPath))
	{
		OutError = FString::Printf(TEXT("Source asset '%s' is missing."), *SourceObjectPath);
		return nullptr;
	}

	UObject* SourceAsset = AssetSubsystem->LoadAsset(SourceObjectPath);
	if (SourceAsset == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to load source asset '%s'."), *SourceObjectPath);
		return nullptr;
	}

	if (!EnsureContentDirectory(AssetSubsystem, GetPackagePathFromObjectPath(ImportedObjectPath), OutError))
	{
		return nullptr;
	}

	if (!RenameLoadedAssetToPath(SourceAsset, ImportedObjectPath, OutError))
	{
		if (!OutError.IsEmpty())
		{
			return nullptr;
		}

		OutError = FString::Printf(TEXT("Failed to move imported asset '%s' to '%s'."), *SourceObjectPath, *ImportedObjectPath);
		return nullptr;
	}

	return AssetSubsystem->LoadAsset(ImportedObjectPath);
}

UObject* EnsureAuthoritativeAssetDuplicate(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& SourceObjectPath,
	const FString& DestinationObjectPath,
	FString& OutError)
{
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
		return nullptr;
	}

	if (AssetSubsystem->DoesAssetExist(DestinationObjectPath))
	{
		return AssetSubsystem->LoadAsset(DestinationObjectPath);
	}

	UObject* SourceAsset = AssetSubsystem->LoadAsset(SourceObjectPath);
	if (SourceAsset == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to load duplicate source asset '%s'."), *SourceObjectPath);
		return nullptr;
	}

	if (!EnsureContentDirectory(AssetSubsystem, GetPackagePathFromObjectPath(DestinationObjectPath), OutError))
	{
		return nullptr;
	}

	UObject* DuplicateAsset = FAssetToolsModule::GetModule().Get().DuplicateAsset(
		GetAssetNameFromObjectPath(DestinationObjectPath),
		GetPackagePathFromObjectPath(DestinationObjectPath),
		SourceAsset);
	if (DuplicateAsset == nullptr)
	{
		OutError = FString::Printf(TEXT("Failed to duplicate asset '%s' to '%s'."), *SourceObjectPath, *DestinationObjectPath);
		return nullptr;
	}

	return DuplicateAsset;
}

bool AssignStaticMeshes(const TArray<UStaticMeshComponent*>& MeshSlots, const TArray<UStaticMesh*>& Meshes)
{
	bool bAssignedAny = false;
	for (int32 Index = 0; Index < MeshSlots.Num(); ++Index)
	{
		if (!IsValid(MeshSlots[Index]))
		{
			continue;
		}

		UStaticMesh* MeshAsset = Meshes.IsValidIndex(Index) ? Meshes[Index] : nullptr;
		MeshSlots[Index]->SetStaticMesh(MeshAsset);
		MeshSlots[Index]->SetRelativeLocation(FVector::ZeroVector);
		MeshSlots[Index]->SetRelativeRotation(FRotator::ZeroRotator);
		MeshSlots[Index]->SetRelativeScale3D(FVector::OneVector);
		bAssignedAny = bAssignedAny || MeshAsset != nullptr;
	}

	return bAssignedAny;
}

UStaticMesh* LoadStaticMeshAsset(UEditorAssetSubsystem* AssetSubsystem, const FString& AssetPath, FString& OutError)
{
	UObject* Asset = LoadAssetChecked(AssetSubsystem, AssetPath, TEXT("StaticMesh"), OutError);
	UStaticMesh* StaticMesh = Cast<UStaticMesh>(Asset);
	if (StaticMesh == nullptr && OutError.IsEmpty())
	{
		OutError = FString::Printf(TEXT("Asset '%s' is not a StaticMesh."), *AssetPath);
	}
	return StaticMesh;
}

void LogStaticMeshMaterialSlots(const UStaticMesh* StaticMesh, const TCHAR* DiagnosticLabel)
{
	if (StaticMesh == nullptr)
	{
		UE_LOG(LogTemp, Warning, TEXT("[%s] LogStaticMeshMaterialSlots: mesh is null."), DiagnosticLabel);
		return;
	}

	const TArray<FStaticMaterial>& Slots = StaticMesh->GetStaticMaterials();
	UE_LOG(LogTemp, Display, TEXT("[%s] Mesh '%s' has %d material slot(s):"), DiagnosticLabel, *StaticMesh->GetPathName(), Slots.Num());
	for (int32 Index = 0; Index < Slots.Num(); ++Index)
	{
		const FStaticMaterial& Slot = Slots[Index];
		const FString MatName = Slot.MaterialInterface != nullptr ? Slot.MaterialInterface->GetPathName() : TEXT("<null>");
		const FString SlotName = Slot.MaterialSlotName.IsNone() ? TEXT("<none>") : Slot.MaterialSlotName.ToString();
		UE_LOG(LogTemp, Display, TEXT("  [%d] SlotName='%s' Material='%s'"), Index, *SlotName, *MatName);
	}
}

bool ApplyStaticMeshMaterials(UStaticMesh* StaticMesh, const TArray<UMaterialInterface*>& Materials, const TCHAR* DiagnosticLabel, FString& OutError)
{
	if (StaticMesh == nullptr)
	{
		OutError = TEXT("Static mesh is null.");
		return false;
	}

	const int32 MeshSlotCount = StaticMesh->GetStaticMaterials().Num();
	if (Materials.Num() != MeshSlotCount)
	{
		UE_LOG(LogTemp, Warning, TEXT("[%s] Material count mismatch for '%s': mesh has %d slot(s), provided %d material(s). Will apply min(%d,%d)."),
			DiagnosticLabel, *StaticMesh->GetPathName(), MeshSlotCount, Materials.Num(), MeshSlotCount, Materials.Num());
	}

	const int32 ApplyCount = FMath::Min(Materials.Num(), MeshSlotCount);
	StaticMesh->Modify();
	for (int32 MaterialIndex = 0; MaterialIndex < ApplyCount; ++MaterialIndex)
	{
		if (Materials[MaterialIndex] == nullptr)
		{
			OutError = FString::Printf(TEXT("[%s] Null material at slot %d for mesh '%s'."), DiagnosticLabel, MaterialIndex, *StaticMesh->GetPathName());
			return false;
		}

		StaticMesh->SetMaterial(MaterialIndex, Materials[MaterialIndex]);
		UE_LOG(LogTemp, Display, TEXT("[%s] Slot %d -> '%s'"), DiagnosticLabel, MaterialIndex, *Materials[MaterialIndex]->GetPathName());
	}

	StaticMesh->PostEditChange();
	StaticMesh->MarkPackageDirty();

	LogStaticMeshMaterialSlots(StaticMesh, DiagnosticLabel);
	return true;
}

struct FBootstrapMaterialFixup
{
	FString MeshAssetPath;
	TArray<FString> MaterialAssetPaths;
	FString DiagnosticLabel;
};

bool ApplyBootstrapMaterialFixups(
	UEditorAssetSubsystem* AssetSubsystem,
	const TArray<FBootstrapMaterialFixup>& Fixups,
	FString& OutError)
{
	for (const FBootstrapMaterialFixup& Fixup : Fixups)
	{
		const TCHAR* Label = *Fixup.DiagnosticLabel;

		UStaticMesh* Mesh = LoadStaticMeshAsset(AssetSubsystem, Fixup.MeshAssetPath, OutError);
		if (Mesh == nullptr)
		{
			OutError = FString::Printf(TEXT("[%s] %s"), Label, *OutError);
			return false;
		}

		UE_LOG(LogTemp, Display, TEXT("[%s] Applying explicit material fixup to '%s':"), Label, *Fixup.MeshAssetPath);
		LogStaticMeshMaterialSlots(Mesh, Label);

		TArray<UMaterialInterface*> Materials;
		Materials.Reserve(Fixup.MaterialAssetPaths.Num());
		for (const FString& MaterialPath : Fixup.MaterialAssetPaths)
		{
			UObject* LoadedAsset = LoadAssetChecked(AssetSubsystem, MaterialPath, TEXT("Material"), OutError);
			if (LoadedAsset == nullptr)
			{
				OutError = FString::Printf(TEXT("[%s] Failed to load material asset '%s': %s"), Label, *MaterialPath, *OutError);
				return false;
			}

			UMaterialInterface* Material = Cast<UMaterialInterface>(LoadedAsset);
			if (Material == nullptr)
			{
				UE_LOG(LogTemp, Warning, TEXT("[%s] Asset '%s' is %s, not a MaterialInterface. Skipping."),
					Label, *MaterialPath, *LoadedAsset->GetClass()->GetName());
				OutError = FString::Printf(TEXT("[%s] Asset '%s' is '%s', not a MaterialInterface."),
					Label, *MaterialPath, *LoadedAsset->GetClass()->GetName());
				return false;
			}

			Materials.Add(Material);
		}

		if (!ApplyStaticMeshMaterials(Mesh, Materials, Label, OutError))
		{
			return false;
		}

		UE_LOG(LogTemp, Display, TEXT("[%s] Material fixup complete."), Label);
	}

	return true;
}

bool ConfigureDynamicVisibleBlueprint(
	UEditorAssetSubsystem* AssetSubsystem,
	UBlueprint* Blueprint,
	const TArray<FString>& MeshAssetPaths,
	const FVector& CollisionExtentCm,
	FString& OutError)
{
	if (Blueprint == nullptr)
	{
		OutError = TEXT("Dynamic visible blueprint is null.");
		return false;
	}

	if (Blueprint->GeneratedClass == nullptr)
	{
		FKismetEditorUtilities::CompileBlueprint(Blueprint);
	}

	AAeroDynamicVisibleActorBase* Defaults = Blueprint->GeneratedClass != nullptr
		? Blueprint->GeneratedClass->GetDefaultObject<AAeroDynamicVisibleActorBase>()
		: nullptr;
	if (Defaults == nullptr)
	{
		OutError = FString::Printf(TEXT("Blueprint '%s' does not generate AAeroDynamicVisibleActorBase defaults."), *Blueprint->GetPathName());
		return false;
	}

	TArray<UStaticMesh*> Meshes;
	for (const FString& MeshAssetPath : MeshAssetPaths)
	{
		if (MeshAssetPath.TrimStartAndEnd().IsEmpty())
		{
			Meshes.Add(nullptr);
			continue;
		}

		UStaticMesh* MeshAsset = LoadStaticMeshAsset(AssetSubsystem, MeshAssetPath, OutError);
		if (MeshAsset == nullptr)
		{
			return false;
		}
		Meshes.Add(MeshAsset);
	}

	Defaults->Modify();
	Defaults->SetCollisionBoxExtentCm(CollisionExtentCm);
	if (!AssignStaticMeshes(Defaults->GetMeshSlots(), Meshes))
	{
		OutError = FString::Printf(TEXT("Blueprint '%s' did not receive any static meshes."), *Blueprint->GetPathName());
		return false;
	}

	Defaults->PostEditChange();
	FBlueprintEditorUtils::MarkBlueprintAsModified(Blueprint);
	Blueprint->MarkPackageDirty();
	return true;
}

bool ConfigureCompositeMeshBlueprint(
	UEditorAssetSubsystem* AssetSubsystem,
	UBlueprint* Blueprint,
	const TArray<FString>& MeshAssetPaths,
	FString& OutError)
{
	if (Blueprint == nullptr)
	{
		OutError = TEXT("Composite mesh blueprint is null.");
		return false;
	}

	if (Blueprint->GeneratedClass == nullptr)
	{
		FKismetEditorUtilities::CompileBlueprint(Blueprint);
	}

	AAeroCompositeMeshActorBase* Defaults = Blueprint->GeneratedClass != nullptr
		? Blueprint->GeneratedClass->GetDefaultObject<AAeroCompositeMeshActorBase>()
		: nullptr;
	if (Defaults == nullptr)
	{
		OutError = FString::Printf(TEXT("Blueprint '%s' does not generate AAeroCompositeMeshActorBase defaults."), *Blueprint->GetPathName());
		return false;
	}

	TArray<UStaticMesh*> Meshes;
	for (const FString& MeshAssetPath : MeshAssetPaths)
	{
		if (MeshAssetPath.TrimStartAndEnd().IsEmpty())
		{
			Meshes.Add(nullptr);
			continue;
		}

		UStaticMesh* MeshAsset = LoadStaticMeshAsset(AssetSubsystem, MeshAssetPath, OutError);
		if (MeshAsset == nullptr)
		{
			return false;
		}
		Meshes.Add(MeshAsset);
	}

	Defaults->Modify();
	if (!AssignStaticMeshes(Defaults->GetMeshSlots(), Meshes))
	{
		OutError = FString::Printf(TEXT("Blueprint '%s' did not receive any static meshes."), *Blueprint->GetPathName());
		return false;
	}

	Defaults->PostEditChange();
	FBlueprintEditorUtils::MarkBlueprintAsModified(Blueprint);
	Blueprint->MarkPackageDirty();
	return true;
}

UBlueprint* CreateOrLoadChildBlueprint(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& AssetObjectPath,
	UClass* ParentClass,
	FString& OutError)
{
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
		return nullptr;
	}

	if (ParentClass == nullptr)
	{
		OutError = FString::Printf(TEXT("Blueprint parent class for '%s' is null."), *AssetObjectPath);
		return nullptr;
	}

	UBlueprint* Blueprint = nullptr;
	if (AssetSubsystem->DoesAssetExist(AssetObjectPath))
	{
		Blueprint = Cast<UBlueprint>(AssetSubsystem->LoadAsset(AssetObjectPath));
		if (Blueprint == nullptr)
		{
			OutError = FString::Printf(TEXT("Existing asset '%s' is not a UBlueprint."), *AssetObjectPath);
			return nullptr;
		}
	}
	else
	{
		UPackage* Package = CreatePackage(*GetPackageNameFromObjectPath(AssetObjectPath));
		if (Package == nullptr)
		{
			OutError = FString::Printf(TEXT("Failed to create package for blueprint '%s'."), *AssetObjectPath);
			return nullptr;
		}

		Blueprint = FKismetEditorUtilities::CreateBlueprint(
			ParentClass,
			Package,
			*GetAssetNameFromObjectPath(AssetObjectPath),
			BPTYPE_Normal,
			UBlueprint::StaticClass(),
			UBlueprintGeneratedClass::StaticClass(),
			FName(AeroBootstrapCallingContext));
		if (Blueprint == nullptr)
		{
			OutError = FString::Printf(TEXT("Failed to create blueprint '%s'."), *AssetObjectPath);
			return nullptr;
		}

		FAssetRegistryModule::AssetCreated(Blueprint);
		Package->MarkPackageDirty();
	}

	if (Blueprint->ParentClass != ParentClass)
	{
		UBlueprintEditorLibrary::ReparentBlueprint(Blueprint, ParentClass);
	}

	FKismetEditorUtilities::CompileBlueprint(Blueprint);
	Blueprint->MarkPackageDirty();
	return Blueprint;
}

template <typename TDataAsset>
TDataAsset* CreateOrLoadTypedDataAsset(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& AssetObjectPath,
	FString& OutError)
{
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
		return nullptr;
	}

	TDataAsset* DataAsset = nullptr;
	if (AssetSubsystem->DoesAssetExist(AssetObjectPath))
	{
		DataAsset = Cast<TDataAsset>(AssetSubsystem->LoadAsset(AssetObjectPath));
		if (DataAsset == nullptr)
		{
			OutError = FString::Printf(
				TEXT("Existing asset '%s' is not a %s."),
				*AssetObjectPath,
				*TDataAsset::StaticClass()->GetName());
			return nullptr;
		}

		return DataAsset;
	}

	UDataAssetFactory* Factory = NewObject<UDataAssetFactory>();
	Factory->DataAssetClass = TDataAsset::StaticClass();

	UObject* CreatedAsset = FAssetToolsModule::GetModule().Get().CreateAsset(
		GetAssetNameFromObjectPath(AssetObjectPath),
		GetPackagePathFromObjectPath(AssetObjectPath),
		TDataAsset::StaticClass(),
		Factory,
		FName(AeroBootstrapCallingContext));
	DataAsset = Cast<TDataAsset>(CreatedAsset);
	if (DataAsset == nullptr)
	{
		OutError = FString::Printf(
			TEXT("Failed to create %s '%s'."),
			*TDataAsset::StaticClass()->GetName(),
			*AssetObjectPath);
		return nullptr;
	}

	return DataAsset;
}

UPedestrianVariantCatalog* CreateOrLoadPedestrianVariantCatalog(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& AssetObjectPath,
	FString& OutError)
{
	return CreateOrLoadTypedDataAsset<UPedestrianVariantCatalog>(AssetSubsystem, AssetObjectPath, OutError);
}

UCrowdAppearancePool* CreateOrLoadCrowdAppearancePool(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& AssetObjectPath,
	FString& OutError)
{
	return CreateOrLoadTypedDataAsset<UCrowdAppearancePool>(AssetSubsystem, AssetObjectPath, OutError);
}

UCrowdRoleProfile* CreateOrLoadCrowdRoleProfile(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& AssetObjectPath,
	FString& OutError)
{
	return CreateOrLoadTypedDataAsset<UCrowdRoleProfile>(AssetSubsystem, AssetObjectPath, OutError);
}

FPedVariantSpec MakePedVariantSpec(
	const TCHAR* VariantId,
	const TCHAR* SkeletalMeshPath,
	float CapsuleRadius = -1.0f,
	float CapsuleHalfHeight = -1.0f,
	float GroundContactOffsetCm = 0.0f,
	const FRotator& MeshRelativeRotation = FRotator::ZeroRotator,
	const FVector& MeshRelativeScale = FVector::OneVector)
{
	FPedVariantSpec Spec;
	Spec.VariantId = FName(VariantId);
	Spec.SkeletalMesh = TSoftObjectPtr<USkeletalMesh>(FSoftObjectPath(SkeletalMeshPath));
	Spec.CapsuleRadius = CapsuleRadius;
	Spec.CapsuleHalfHeight = CapsuleHalfHeight;
	Spec.MeshRelativeRotation = MeshRelativeRotation;
	Spec.MeshRelativeScale = MeshRelativeScale;
	Spec.GroundContactOffsetCm = GroundContactOffsetCm;
	Spec.ObserveMontageOverride = TSoftObjectPtr<UAnimMontage>(FSoftObjectPath(PedObserveMontagePath));
	Spec.StartCrossMontageOverride = TSoftObjectPtr<UAnimMontage>(FSoftObjectPath(PedStartCrossMontagePath));
	return Spec;
}

TArray<FPedVariantSpec> BuildAuthoritativePedVariants()
{
	TArray<FPedVariantSpec> Variants;
	Variants.Reserve(4);
	Variants.Add(MakePedVariantSpec(
		TEXT("adult_male_commuter"),
		TEXT("/Game/MixamoAssets/Characters/SK_SRC_AdultMale_01.SK_SRC_AdultMale_01"),
		-1.0f,
		-1.0f,
		0.0f,
		FRotator(0.0f, -90.0f, 0.0f),
		FVector(0.5f, 0.5f, 0.5f)));
	Variants.Add(MakePedVariantSpec(
		TEXT("adult_female_commuter"),
		TEXT("/Game/MixamoAssets/Characters/SK_SRC_AdultFemale_01.SK_SRC_AdultFemale_01"),
		-1.0f,
		-1.0f,
		0.0f,
		FRotator(0.0f, -90.0f, 0.0f)));
	Variants.Add(MakePedVariantSpec(
		TEXT("child_crossing"),
		TEXT("/Game/MixamoAssets/Characters/SK_SRC_Child_01.SK_SRC_Child_01"),
		20.0f,
		44.0f,
		0.0f,
		FRotator(0.0f, -90.0f, 0.0f)));
	Variants.Add(MakePedVariantSpec(
		TEXT("elder_observer"),
		TEXT("/Game/MixamoAssets/Characters/SK_SRC_Elder_01.SK_SRC_Elder_01"),
		-1.0f,
		-1.0f,
		0.0f,
		FRotator(0.0f, -90.0f, 0.0f)));
	return Variants;
}

bool UpdatePedestrianVariantCatalog(UPedestrianVariantCatalog* Catalog, FString& OutError)
{
	if (Catalog == nullptr)
	{
		OutError = TEXT("Pedestrian variant catalog is null.");
		return false;
	}

	Catalog->Modify();
	Catalog->Variants = BuildAuthoritativePedVariants();
	Catalog->PostEditChange();
	Catalog->MarkPackageDirty();
	return true;
}

FCrowdAccessorySpec MakeCrowdAccessorySpec(
	const TCHAR* AccessoryTag,
	const TCHAR* MeshPath,
	const TCHAR* SocketName,
	float Probability,
	const FVector& RelativeLocation = FVector::ZeroVector,
	const FRotator& RelativeRotation = FRotator::ZeroRotator,
	const FVector& RelativeScale = FVector::OneVector)
{
	FCrowdAccessorySpec Spec;
	Spec.AccessoryTag = FName(AccessoryTag);
	Spec.Mesh = TSoftObjectPtr<UStaticMesh>(FSoftObjectPath(MeshPath));
	Spec.SocketName = FName(SocketName);
	Spec.Probability = Probability;
	Spec.RelativeLocation = RelativeLocation;
	Spec.RelativeRotation = RelativeRotation;
	Spec.RelativeScale = RelativeScale;
	return Spec;
}

void AddGroupedAccessorySpecs(
	TArray<FCrowdAccessorySpec>& OutAccessories,
	const TCHAR* AccessoryTag,
	std::initializer_list<const TCHAR*> MeshPaths,
	const TCHAR* SocketName,
	float Probability,
	const FVector& RelativeLocation = FVector::ZeroVector,
	const FRotator& RelativeRotation = FRotator::ZeroRotator,
	const FVector& RelativeScale = FVector::OneVector)
{
	for (const TCHAR* MeshPath : MeshPaths)
	{
		OutAccessories.Add(MakeCrowdAccessorySpec(AccessoryTag, MeshPath, SocketName, Probability, RelativeLocation, RelativeRotation, RelativeScale));
	}
}

FCrowdAppearanceEntry MakeCrowdAppearanceEntry(
	const TCHAR* AppearanceId,
	ECrowdGender Gender,
	ECrowdAgeGroup AgeGroup,
	const FVector2D& ScaleRange,
	std::initializer_list<const TCHAR*> SpawnTags,
	std::initializer_list<const TCHAR*> AccessoryTags,
	const TArray<FCrowdAccessorySpec>& OptionalAccessories)
{
	FCrowdAppearanceEntry Entry;
	Entry.AppearanceId = FName(AppearanceId);
	Entry.VariantId = FName(AppearanceId);
	Entry.Gender = Gender;
	Entry.AgeGroup = AgeGroup;
	Entry.Weight = 1.0f;
	Entry.ScaleRange = ScaleRange;
	Entry.OptionalAccessories = OptionalAccessories;

	for (const TCHAR* Tag : SpawnTags)
	{
		Entry.SpawnTags.Add(FName(Tag));
	}

	for (const TCHAR* Tag : AccessoryTags)
	{
		Entry.AccessoryTags.Add(FName(Tag));
	}

	return Entry;
}

TArray<FCrowdAppearanceEntry> BuildAuthoritativeCrowdAppearances()
{
	TArray<FCrowdAppearanceEntry> Entries;
	Entries.Reserve(4);

	TArray<FCrowdAccessorySpec> AdultMaleAccessories;
	AddGroupedAccessorySpecs(
		AdultMaleAccessories,
		TEXT("backpack"),
		{
			BackpackPrimaryImportedPath,
			BackpackExtrasImportedPath,
			BackpackRespaldoImportedPath,
			BackpackRespaldo2ImportedPath,
			BackpackStrapAdjustImportedPath,
			BackpackStrapBackImportedPath
		},
		TEXT("mixamorig:Spine2"),
		0.30f);
	AdultMaleAccessories.Add(MakeCrowdAccessorySpec(
		TEXT("phone"),
		PhonePrimaryImportedPath,
		TEXT("mixamorig:RightHand"),
		0.20f));
	Entries.Add(MakeCrowdAppearanceEntry(
		TEXT("adult_male_commuter"),
		ECrowdGender::Male,
		ECrowdAgeGroup::Adult,
		FVector2D(0.50f, 0.50f),
		{TEXT("pedestrian"), TEXT("cityops"), TEXT("adult"), TEXT("male"), TEXT("commuter")},
		{TEXT("backpack"), TEXT("phone")},
		AdultMaleAccessories));

	TArray<FCrowdAccessorySpec> AdultFemaleAccessories;
	AdultFemaleAccessories.Add(MakeCrowdAccessorySpec(
		TEXT("phone"),
		PhonePrimaryImportedPath,
		TEXT("mixamorig:RightHand"),
		0.35f));
	AddGroupedAccessorySpecs(
		AdultFemaleAccessories,
		TEXT("umbrella"),
		{
			UmbrellaPrimaryImportedPath,
			UmbrellaPart01ImportedPath,
			UmbrellaPart02ImportedPath
		},
		TEXT("mixamorig:RightHand"),
		0.20f,
		FVector(0.0f, 0.0f, 90.0f),
		FRotator::ZeroRotator,
		FVector(0.5f, 0.5f, 0.5f));
	Entries.Add(MakeCrowdAppearanceEntry(
		TEXT("adult_female_commuter"),
		ECrowdGender::Female,
		ECrowdAgeGroup::Adult,
		FVector2D(1.0f, 1.0f),
		{TEXT("pedestrian"), TEXT("cityops"), TEXT("adult"), TEXT("female"), TEXT("commuter")},
		{TEXT("phone"), TEXT("umbrella")},
		AdultFemaleAccessories));

	Entries.Add(MakeCrowdAppearanceEntry(
		TEXT("child_crossing"),
		ECrowdGender::Unknown,
		ECrowdAgeGroup::Child,
		FVector2D(0.72f, 0.80f),
		{TEXT("pedestrian"), TEXT("cityops"), TEXT("child"), TEXT("crossing")},
		{},
		TArray<FCrowdAccessorySpec>()));

	TArray<FCrowdAccessorySpec> ElderAccessories;
	AddGroupedAccessorySpecs(
		ElderAccessories,
		TEXT("umbrella"),
		{
			UmbrellaPrimaryImportedPath,
			UmbrellaPart01ImportedPath,
			UmbrellaPart02ImportedPath
		},
		TEXT("mixamorig:RightHand"),
		0.50f,
		FVector(0.0f, 0.0f, 90.0f),
		FRotator::ZeroRotator,
		FVector(0.5f, 0.5f, 0.5f));
	Entries.Add(MakeCrowdAppearanceEntry(
		TEXT("elder_observer"),
		ECrowdGender::Male,
		ECrowdAgeGroup::Elder,
		FVector2D(0.97f, 0.97f),
		{TEXT("pedestrian"), TEXT("cityops"), TEXT("elder"), TEXT("observer")},
		{TEXT("umbrella")},
		ElderAccessories));

	return Entries;
}

bool UpdateCrowdAppearancePool(UCrowdAppearancePool* AppearancePool, FString& OutError)
{
	if (AppearancePool == nullptr)
	{
		OutError = TEXT("Crowd appearance pool is null.");
		return false;
	}

	AppearancePool->Modify();
	AppearancePool->Entries = BuildAuthoritativeCrowdAppearances();
	AppearancePool->PostEditChange();
	AppearancePool->MarkPackageDirty();
	return true;
}

bool UpdateCrowdRoleProfile(UCrowdRoleProfile* RoleProfile, FString& OutError)
{
	if (RoleProfile == nullptr)
	{
		OutError = TEXT("Crowd role profile is null.");
		return false;
	}

	RoleProfile->Modify();
	RoleProfile->AllowedGenders.Reset();
	RoleProfile->AllowedAgeGroups.Reset();
	RoleProfile->RequiredTags.Reset();
	RoleProfile->BlockedTags.Reset();
	RoleProfile->WeightMultipliers.Reset();
	RoleProfile->CountOverride = -1;
	RoleProfile->DefaultBehaviorMode = FName(TEXT("idle"));
	RoleProfile->DefaultSpawnRadius = 1200.0f;
	RoleProfile->DefaultMinSpacing = 120.0f;
	RoleProfile->PostEditChange();
	RoleProfile->MarkPackageDirty();
	return true;
}

bool ConfigurePedestrianBlueprintDefaults(
	UEditorAssetSubsystem* AssetSubsystem,
	UBlueprint* Blueprint,
	UPedestrianVariantCatalog* Catalog,
	FString& OutError)
{
	if (Blueprint == nullptr || Catalog == nullptr)
	{
		OutError = TEXT("Pedestrian blueprint or catalog is null.");
		return false;
	}

	if (Blueprint->GeneratedClass == nullptr)
	{
		FKismetEditorUtilities::CompileBlueprint(Blueprint);
	}

	APedestrianCharacter* PedestrianDefaults = Blueprint->GeneratedClass != nullptr
		? Blueprint->GeneratedClass->GetDefaultObject<APedestrianCharacter>()
		: nullptr;
	if (PedestrianDefaults == nullptr)
	{
		OutError = FString::Printf(TEXT("Blueprint '%s' does not generate APedestrianCharacter defaults."), *Blueprint->GetPathName());
		return false;
	}

	UAnimMontage* ObserveMontage = Cast<UAnimMontage>(LoadAssetChecked(AssetSubsystem, PedObserveMontagePath, TEXT("ObserveMontage"), OutError));
	if (ObserveMontage == nullptr)
	{
		return false;
	}

	UAnimMontage* StartCrossMontage = Cast<UAnimMontage>(LoadAssetChecked(AssetSubsystem, PedStartCrossMontagePath, TEXT("StartCrossMontage"), OutError));
	if (StartCrossMontage == nullptr)
	{
		return false;
	}

	PedestrianDefaults->Modify();
	PedestrianDefaults->VariantCatalog = Catalog;
	PedestrianDefaults->InitialVariantId = FName(DefaultPedSpawnVariantId);
	PedestrianDefaults->ObserveMontage = ObserveMontage;
	PedestrianDefaults->StartCrossMontage = StartCrossMontage;
	PedestrianDefaults->PostEditChange();

	FBlueprintEditorUtils::MarkBlueprintAsModified(Blueprint);
	Blueprint->MarkPackageDirty();
	return true;
}

bool SaveAssets(UEditorAssetSubsystem* AssetSubsystem, const TArray<UObject*>& AssetsToSave, FString& OutError)
{
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("EditorAssetSubsystem is unavailable.");
		return false;
	}

	if (!AssetSubsystem->SaveLoadedAssets(AssetsToSave, false))
	{
		OutError = TEXT("Failed to save generated AeroWorldContent assets.");
		return false;
	}

	return true;
}

bool ValidateBlueprintParent(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& BlueprintAssetPath,
	UClass* ExpectedParentClass,
	TArray<FString>& OutErrors)
{
	UBlueprint* Blueprint = AssetSubsystem != nullptr ? Cast<UBlueprint>(AssetSubsystem->LoadAsset(BlueprintAssetPath)) : nullptr;
	if (Blueprint == nullptr)
	{
		OutErrors.Add(FString::Printf(TEXT("Blueprint '%s' is missing or failed to load."), *BlueprintAssetPath));
		return false;
	}

	if (ExpectedParentClass == nullptr)
	{
		OutErrors.Add(FString::Printf(TEXT("Expected parent class for '%s' is null."), *BlueprintAssetPath));
		return false;
	}

	if (Blueprint->ParentClass != ExpectedParentClass)
	{
		OutErrors.Add(
			FString::Printf(
				TEXT("Blueprint '%s' parent mismatch. Expected '%s', got '%s'."),
				*BlueprintAssetPath,
				*ExpectedParentClass->GetPathName(),
				Blueprint->ParentClass != nullptr ? *Blueprint->ParentClass->GetPathName() : TEXT("<null>")));
		return false;
	}

	return true;
}

bool ValidateTriggerInheritance(UEditorAssetSubsystem* AssetSubsystem, const FString& BlueprintAssetPath, TArray<FString>& OutErrors)
{
	UBlueprint* Blueprint = AssetSubsystem != nullptr ? Cast<UBlueprint>(AssetSubsystem->LoadAsset(BlueprintAssetPath)) : nullptr;
	if (Blueprint == nullptr || Blueprint->GeneratedClass == nullptr)
	{
		OutErrors.Add(FString::Printf(TEXT("Trigger blueprint '%s' is missing, failed to load, or is not compiled."), *BlueprintAssetPath));
		return false;
	}

	if (!Blueprint->GeneratedClass->IsChildOf(AAeroTriggerZoneBase::StaticClass()))
	{
		OutErrors.Add(FString::Printf(TEXT("Trigger blueprint '%s' does not inherit from AAeroTriggerZoneBase."), *BlueprintAssetPath));
		return false;
	}

	return true;
}

bool ValidateDynamicVisibleMeshes(
	UEditorAssetSubsystem* AssetSubsystem,
	const FString& BlueprintAssetPath,
	int32 MinimumAssignedMeshes,
	TArray<FString>& OutErrors)
{
	UBlueprint* Blueprint = AssetSubsystem != nullptr ? Cast<UBlueprint>(AssetSubsystem->LoadAsset(BlueprintAssetPath)) : nullptr;
	if (Blueprint == nullptr || Blueprint->GeneratedClass == nullptr)
	{
		OutErrors.Add(FString::Printf(TEXT("Dynamic blueprint '%s' is missing or not compiled."), *BlueprintAssetPath));
		return false;
	}

	AAeroDynamicVisibleActorBase* Defaults = Blueprint->GeneratedClass->GetDefaultObject<AAeroDynamicVisibleActorBase>();
	if (Defaults == nullptr)
	{
		OutErrors.Add(FString::Printf(TEXT("Dynamic blueprint '%s' does not resolve AAeroDynamicVisibleActorBase defaults."), *BlueprintAssetPath));
		return false;
	}

	int32 AssignedMeshCount = 0;
	for (UStaticMeshComponent* MeshComponent : Defaults->GetMeshSlots())
	{
		if (IsValid(MeshComponent) && MeshComponent->GetStaticMesh() != nullptr)
		{
			++AssignedMeshCount;
		}
	}

	if (AssignedMeshCount < MinimumAssignedMeshes)
	{
		OutErrors.Add(
			FString::Printf(
				TEXT("Dynamic blueprint '%s' expected at least %d assigned mesh slots but found %d."),
				*BlueprintAssetPath,
				MinimumAssignedMeshes,
				AssignedMeshCount));
		return false;
	}

	return true;
}

bool ValidatePedCatalog(UEditorAssetSubsystem* AssetSubsystem, TArray<FString>& OutErrors)
{
	UPedestrianVariantCatalog* Catalog = AssetSubsystem != nullptr ? Cast<UPedestrianVariantCatalog>(AssetSubsystem->LoadAsset(PedCatalogAssetPath)) : nullptr;
	if (Catalog == nullptr)
	{
		OutErrors.Add(FString::Printf(TEXT("Pedestrian variant catalog '%s' is missing or failed to load."), PedCatalogAssetPath));
		return false;
	}

	const TArray<FName> ExpectedIds = {
		FName(TEXT("adult_male_commuter")),
		FName(TEXT("adult_female_commuter")),
		FName(TEXT("child_crossing")),
		FName(TEXT("elder_observer"))};

	bool bAllFound = true;
	for (const FName VariantId : ExpectedIds)
	{
		FPedVariantSpec VariantSpec;
		if (!Catalog->FindVariantById(VariantId, VariantSpec))
		{
			OutErrors.Add(FString::Printf(TEXT("Pedestrian variant catalog is missing variant_id '%s'."), *VariantId.ToString()));
			bAllFound = false;
		}
	}

	if (Catalog->Variants.Num() != ExpectedIds.Num())
	{
		OutErrors.Add(
			FString::Printf(
				TEXT("Pedestrian variant catalog expected %d variants but found %d."),
				ExpectedIds.Num(),
				Catalog->Variants.Num()));
		bAllFound = false;
	}

	return bAllFound;
}

bool ValidateCrowdAppearancePool(UEditorAssetSubsystem* AssetSubsystem, TArray<FString>& OutErrors)
{
	UCrowdAppearancePool* AppearancePool = AssetSubsystem != nullptr ? Cast<UCrowdAppearancePool>(AssetSubsystem->LoadAsset(CrowdAppearancePoolAssetPath)) : nullptr;
	if (AppearancePool == nullptr)
	{
		OutErrors.Add(FString::Printf(TEXT("Crowd appearance pool '%s' is missing or failed to load."), CrowdAppearancePoolAssetPath));
		return false;
	}

	struct FExpectedAppearance
	{
		FName AppearanceId;
		FName VariantId;
		ECrowdGender Gender;
		ECrowdAgeGroup AgeGroup;
		TArray<FName> SpawnTags;
		int32 MinimumAccessorySpecs = 0;
	};

	TArray<FExpectedAppearance> ExpectedAppearances;
	ExpectedAppearances.Reserve(4);

	FExpectedAppearance AdultMale;
	AdultMale.AppearanceId = FName(TEXT("adult_male_commuter"));
	AdultMale.VariantId = FName(TEXT("adult_male_commuter"));
	AdultMale.Gender = ECrowdGender::Male;
	AdultMale.AgeGroup = ECrowdAgeGroup::Adult;
	AdultMale.SpawnTags = {FName(TEXT("pedestrian")), FName(TEXT("cityops")), FName(TEXT("adult")), FName(TEXT("male")), FName(TEXT("commuter"))};
	AdultMale.MinimumAccessorySpecs = 1;
	ExpectedAppearances.Add(AdultMale);

	FExpectedAppearance AdultFemale;
	AdultFemale.AppearanceId = FName(TEXT("adult_female_commuter"));
	AdultFemale.VariantId = FName(TEXT("adult_female_commuter"));
	AdultFemale.Gender = ECrowdGender::Female;
	AdultFemale.AgeGroup = ECrowdAgeGroup::Adult;
	AdultFemale.SpawnTags = {FName(TEXT("pedestrian")), FName(TEXT("cityops")), FName(TEXT("adult")), FName(TEXT("female")), FName(TEXT("commuter"))};
	AdultFemale.MinimumAccessorySpecs = 1;
	ExpectedAppearances.Add(AdultFemale);

	FExpectedAppearance Child;
	Child.AppearanceId = FName(TEXT("child_crossing"));
	Child.VariantId = FName(TEXT("child_crossing"));
	Child.Gender = ECrowdGender::Unknown;
	Child.AgeGroup = ECrowdAgeGroup::Child;
	Child.SpawnTags = {FName(TEXT("pedestrian")), FName(TEXT("cityops")), FName(TEXT("child")), FName(TEXT("crossing"))};
	ExpectedAppearances.Add(Child);

	FExpectedAppearance Elder;
	Elder.AppearanceId = FName(TEXT("elder_observer"));
	Elder.VariantId = FName(TEXT("elder_observer"));
	Elder.Gender = ECrowdGender::Male;
	Elder.AgeGroup = ECrowdAgeGroup::Elder;
	Elder.SpawnTags = {FName(TEXT("pedestrian")), FName(TEXT("cityops")), FName(TEXT("elder")), FName(TEXT("observer"))};
	Elder.MinimumAccessorySpecs = 1;
	ExpectedAppearances.Add(Elder);

	bool bAllFound = true;
	for (const FExpectedAppearance& Expected : ExpectedAppearances)
	{
		const FCrowdAppearanceEntry* Entry = AppearancePool->Entries.FindByPredicate(
			[&Expected](const FCrowdAppearanceEntry& Candidate)
			{
				return Candidate.AppearanceId == Expected.AppearanceId;
			});
		if (Entry == nullptr)
		{
			OutErrors.Add(FString::Printf(TEXT("Crowd appearance pool is missing appearance_id '%s'."), *Expected.AppearanceId.ToString()));
			bAllFound = false;
			continue;
		}

		if (Entry->VariantId != Expected.VariantId)
		{
			OutErrors.Add(FString::Printf(TEXT("Crowd appearance '%s' expected variant_id '%s' but found '%s'."), *Expected.AppearanceId.ToString(), *Expected.VariantId.ToString(), *Entry->VariantId.ToString()));
			bAllFound = false;
		}

		if (Entry->Gender != Expected.Gender || Entry->AgeGroup != Expected.AgeGroup)
		{
			OutErrors.Add(FString::Printf(TEXT("Crowd appearance '%s' has unexpected gender/age metadata."), *Expected.AppearanceId.ToString()));
			bAllFound = false;
		}

		if (Entry->Weight <= 0.0f || Entry->ScaleRange.X <= 0.0f || Entry->ScaleRange.Y <= 0.0f)
		{
			OutErrors.Add(FString::Printf(TEXT("Crowd appearance '%s' has invalid weight or scale range."), *Expected.AppearanceId.ToString()));
			bAllFound = false;
		}

		for (const FName& Tag : Expected.SpawnTags)
		{
			if (!Entry->SpawnTags.Contains(Tag))
			{
				OutErrors.Add(FString::Printf(TEXT("Crowd appearance '%s' is missing spawn tag '%s'."), *Expected.AppearanceId.ToString(), *Tag.ToString()));
				bAllFound = false;
			}
		}

		if (Entry->OptionalAccessories.Num() < Expected.MinimumAccessorySpecs)
		{
			OutErrors.Add(FString::Printf(TEXT("Crowd appearance '%s' expected at least %d accessory specs but found %d."), *Expected.AppearanceId.ToString(), Expected.MinimumAccessorySpecs, Entry->OptionalAccessories.Num()));
			bAllFound = false;
		}
	}

	if (AppearancePool->Entries.Num() != ExpectedAppearances.Num())
	{
		OutErrors.Add(FString::Printf(TEXT("Crowd appearance pool expected %d entries but found %d."), ExpectedAppearances.Num(), AppearancePool->Entries.Num()));
		bAllFound = false;
	}

	return bAllFound;
}

bool ValidateCrowdRoleProfile(UEditorAssetSubsystem* AssetSubsystem, TArray<FString>& OutErrors)
{
	UCrowdRoleProfile* RoleProfile = AssetSubsystem != nullptr ? Cast<UCrowdRoleProfile>(AssetSubsystem->LoadAsset(CrowdRoleProfileAssetPath)) : nullptr;
	if (RoleProfile == nullptr)
	{
		OutErrors.Add(FString::Printf(TEXT("Crowd role profile '%s' is missing or failed to load."), CrowdRoleProfileAssetPath));
		return false;
	}

	bool bValid = true;
	if (RoleProfile->AllowedGenders.Num() != 0 || RoleProfile->AllowedAgeGroups.Num() != 0 || RoleProfile->RequiredTags.Num() != 0 || RoleProfile->WeightMultipliers.Num() != 0)
	{
		OutErrors.Add(TEXT("Crowd role profile default filters (AllowedGenders, AllowedAgeGroups, RequiredTags, WeightMultipliers) are expected to be empty."));
		bValid = false;
	}

	if (RoleProfile->BlockedTags.Num() != 0)
	{
		OutErrors.Add(TEXT("Crowd role profile BlockedTags expected to be empty."));
		bValid = false;
	}

	if (RoleProfile->CountOverride != -1)
	{
		OutErrors.Add(TEXT("Crowd role profile CountOverride expected -1."));
		bValid = false;
	}

	if (RoleProfile->DefaultBehaviorMode != FName(TEXT("idle")))
	{
		OutErrors.Add(TEXT("Crowd role profile DefaultBehaviorMode expected 'idle'."));
		bValid = false;
	}

	if (!FMath::IsNearlyEqual(RoleProfile->DefaultSpawnRadius, 1200.0f) || !FMath::IsNearlyEqual(RoleProfile->DefaultMinSpacing, 120.0f))
	{
		OutErrors.Add(TEXT("Crowd role profile spawn defaults expected radius=1200 and min_spacing=120."));
		bValid = false;
	}

	return bValid;
}

bool ValidatePedestrianRuntimeSettingsDefaults(TArray<FString>& OutErrors)
{
	const UPedestrianRuntimeSettings* Settings = GetDefault<UPedestrianRuntimeSettings>();
	if (Settings == nullptr)
	{
		OutErrors.Add(TEXT("PedestrianRuntimeSettings defaults are unavailable."));
		return false;
	}

	bool bValid = true;
	if (Settings->DefaultPedestrianClass.LoadSynchronous() == nullptr)
	{
		OutErrors.Add(TEXT("PedestrianRuntimeSettings DefaultPedestrianClass failed to resolve."));
		bValid = false;
	}

	if (Settings->DefaultSpawnVariantId != FName(DefaultPedSpawnVariantId))
	{
		OutErrors.Add(TEXT("PedestrianRuntimeSettings DefaultSpawnVariantId expected 'adult_female_commuter'."));
		bValid = false;
	}

	if (Settings->DefaultCrowdAppearancePool.LoadSynchronous() == nullptr)
	{
		OutErrors.Add(TEXT("PedestrianRuntimeSettings DefaultCrowdAppearancePool failed to resolve."));
		bValid = false;
	}

	if (Settings->DefaultCrowdRoleProfile.LoadSynchronous() == nullptr)
	{
		OutErrors.Add(TEXT("PedestrianRuntimeSettings DefaultCrowdRoleProfile failed to resolve."));
		bValid = false;
	}

	return bValid;
}

bool ValidateAssetCatalogMappings(UEditorAssetSubsystem* AssetSubsystem, TArray<FString>& OutErrors)
{
	TSharedPtr<FJsonObject> RootObject;
	FString LoadError;
	if (!LoadJsonObjectFromFile(GetAssetCatalogPath(), RootObject, LoadError))
	{
		OutErrors.Add(LoadError);
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Assets = nullptr;
	if (!RootObject->TryGetArrayField(TEXT("assets"), Assets) || Assets == nullptr)
	{
		OutErrors.Add(TEXT("asset_catalog.json is missing the 'assets' array."));
		return false;
	}

	bool bAllValid = true;
	for (const TSharedPtr<FJsonValue>& AssetValue : *Assets)
	{
		const TSharedPtr<FJsonObject> AssetObject = AssetValue.IsValid() ? AssetValue->AsObject() : nullptr;
		if (!AssetObject.IsValid())
		{
			OutErrors.Add(TEXT("asset_catalog.json contains a non-object asset entry."));
			bAllValid = false;
			continue;
		}

		FString LogicalAssetId;
		FString AssetPath;
		AssetObject->TryGetStringField(TEXT("logical_asset_id"), LogicalAssetId);
		AssetObject->TryGetStringField(TEXT("ue_asset_path"), AssetPath);
		if (AssetPath.IsEmpty())
		{
			continue;
		}
		if (AssetPath.StartsWith(TEXT("/Engine/")))
		{
			continue;
		}

		FSoftObjectPath SoftObjectPath(AssetPath);
		UObject* ResolvedAsset = SoftObjectPath.ResolveObject();
		if (ResolvedAsset == nullptr)
		{
			ResolvedAsset = SoftObjectPath.TryLoad();
		}

		bool bAssetExists = ResolvedAsset != nullptr;
		if (!bAssetExists)
		{
			const FString PackageName = FPackageName::ObjectPathToPackageName(AssetPath);
			FString PackageFilename;
			bAssetExists = !PackageName.IsEmpty() && FPackageName::DoesPackageExist(PackageName, &PackageFilename);
		}

		if (!bAssetExists)
		{
			OutErrors.Add(
				FString::Printf(
					TEXT("asset_catalog.json path for logical_asset_id '%s' does not exist: %s"),
					*LogicalAssetId,
					*AssetPath));
			bAllValid = false;
		}
	}

	return bAllValid;
}
} // namespace

bool UAeroEditorToolsSubsystem::CompilePedSemanticBundleForMap(const FString& MapId, FString& OutError)
{
	UWorld* EditorWorld = GEditor != nullptr ? GEditor->GetEditorWorldContext().World() : nullptr;
	UObject* CompilerOuter = EditorWorld != nullptr ? static_cast<UObject*>(EditorWorld) : static_cast<UObject*>(this);
	UAeroPedNavSemanticSubsystem* Compiler = NewObject<UAeroPedNavSemanticSubsystem>(CompilerOuter);
	if (Compiler == nullptr)
	{
		OutError = TEXT("Failed to allocate ped semantic compiler.");
		return false;
	}

	const FString MapDir = GetMapConfigDir(MapId);
	TSharedPtr<FJsonObject> MapContext;
	const FString MapContextPath = FPaths::Combine(MapDir, TEXT("map_context.json"));
	if (FPaths::FileExists(MapContextPath) && LoadJsonObjectFromFile(MapContextPath, MapContext, OutError))
	{
		Compiler->SetMapContext(MapId, MapContext);
	}
	else if (FPaths::FileExists(MapContextPath) && !OutError.IsEmpty())
	{
		return false;
	}

	const FString SourcePath = FPaths::Combine(MapDir, TEXT("ped_nav_semantic.source.json"));
	const FString BundlePath = FPaths::Combine(MapDir, TEXT("ped_nav_semantic.bundle.json"));
	return Compiler->CompileSemanticBundle(SourcePath, BundlePath, OutError);
}

bool UAeroEditorToolsSubsystem::BootstrapPedSemanticSourceForMap(const FString& MapId, FString& OutError)
{
	const FString MapDir = GetMapConfigDir(MapId);
	const FString SourcePath = FPaths::Combine(MapDir, TEXT("ped_nav_semantic.source.json"));
	if (FPaths::FileExists(SourcePath))
	{
		return true;
	}

	TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
	Root->SetArrayField(TEXT("sidewalk_segments"), TArray<TSharedPtr<FJsonValue>>());
	Root->SetArrayField(TEXT("crossing_connectors"), TArray<TSharedPtr<FJsonValue>>());
	Root->SetArrayField(TEXT("waiting_zones"), TArray<TSharedPtr<FJsonValue>>());

	TSharedPtr<FJsonObject> ProjectionRules = MakeShared<FJsonObject>();
	ProjectionRules->SetNumberField(TEXT("max_snap_distance_m"), 6.0);
	ProjectionRules->SetNumberField(TEXT("ground_trace_half_height_m"), 25.0);
	Root->SetObjectField(TEXT("projection_rules"), ProjectionRules);

	FString Output;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
	if (!FJsonSerializer::Serialize(Root.ToSharedRef(), Writer))
	{
		OutError = FString::Printf(TEXT("Failed to serialize bootstrap ped semantic source for map '%s'."), *MapId);
		return false;
	}

	if (!FFileHelper::SaveStringToFile(Output, *SourcePath))
	{
		OutError = FString::Printf(TEXT("Failed to write bootstrap ped semantic source: %s"), *SourcePath);
		return false;
	}

	return true;
}

bool UAeroEditorToolsSubsystem::BootstrapAeroWorldContentAssets(FString& OutError)
{
	UEditorAssetSubsystem* AssetSubsystem = ResolveEditorAssetSubsystem(OutError);
	if (AssetSubsystem == nullptr)
	{
		return false;
	}

	const TArray<FString> Directories = {
		AeroWorldContentRoot,
		AeroWorldContentBlueprintRoot,
		AeroWorldContentTriggerDir,
		AeroWorldContentPedestrianDir,
		AeroWorldContentPropDir,
		AeroWorldContentVehicleDir,
		AeroWorldContentUavDir,
		AeroWorldContentDataAssetDir,
		AeroWorldContentPedDataDir,
		AeroWorldContentCrowdDataDir,
		ChargerMeshDir,
		FString::Printf(TEXT("%s/_Imported"), ChargerMeshDir),
		LandingPadMeshDir,
		FString::Printf(TEXT("%s/_Imported"), LandingPadMeshDir),
		RadioMeshDir,
		FString::Printf(TEXT("%s/_Imported"), RadioMeshDir),
		TrafficControlMeshDir,
		FString::Printf(TEXT("%s/_Imported"), TrafficControlMeshDir),
		RoadworkMeshDir,
		FString::Printf(TEXT("%s/_Imported"), RoadworkMeshDir),
		ServiceMeshDir,
		FString::Printf(TEXT("%s/_Imported"), ServiceMeshDir),
		MiscMeshDir,
		FString::Printf(TEXT("%s/_Imported"), MiscMeshDir),
		EmergencyVehicleMeshDir,
		FString::Printf(TEXT("%s/_Imported"), EmergencyVehicleMeshDir)};
	for (const FString& DirectoryPath : Directories)
	{
		if (!EnsureContentDirectory(AssetSubsystem, DirectoryPath, OutError))
		{
			return false;
		}
	}

	UClass* PedParentClass = LoadParentBlueprintClass(AssetSubsystem, PedBlueprintParentPath, OutError);
	if (PedParentClass == nullptr)
	{
		return false;
	}

	UClass* VehicleServiceParentClass = LoadParentBlueprintClass(AssetSubsystem, VehicleServiceParentPath, OutError);
	if (VehicleServiceParentClass == nullptr)
	{
		return false;
	}

	UClass* UavInspectionParentClass = LoadParentBlueprintClass(AssetSubsystem, UavInspectionParentPath, OutError);
	if (UavInspectionParentClass == nullptr)
	{
		return false;
	}

	const TArray<TPair<FString, FString>> ImportedAssetMoves = {
		{ChargerMeshSourcePath, ChargerMeshImportedPath},
		{LandingPadMeshSourcePath, LandingPadMeshImportedPath},
		{RadioMeshSourcePath, RadioMeshImportedPath},
		{PoliceSignMeshSourcePath, PoliceSignMeshImportedPath},
		{TrafficLightMeshSourcePath, TrafficLightMeshImportedPath},
		{PoliceTapeMeshSourcePath, PoliceTapeMeshImportedPath},
		{TrafficConeMeshSourcePath, TrafficConeMeshImportedPath},
		{BarrierMeshSourcePath, BarrierMeshImportedPath},
		{DeliveryBagMeshSourcePath, DeliveryBagMeshImportedPath},
		{AmbulanceMeshSourcePath, AmbulanceMeshImportedPath},
		{PoliceBodySourcePath, PoliceBodyImportedPath},
		{PoliceGlassSourcePath, PoliceGlassImportedPath},
		{PoliceInteriorSourcePath, PoliceInteriorImportedPath},
		{PoliceShadowSourcePath, PoliceShadowImportedPath},
		{PoliceWheelFlSourcePath, PoliceWheelFlImportedPath},
		{PoliceWheelFrSourcePath, PoliceWheelFrImportedPath},
		{PoliceWheelRlSourcePath, PoliceWheelRlImportedPath},
		{PoliceWheelRrSourcePath, PoliceWheelRrImportedPath},
		{BackpackPrimarySourcePath, BackpackPrimaryImportedPath},
		{BackpackExtrasSourcePath, BackpackExtrasImportedPath},
		{BackpackRespaldoSourcePath, BackpackRespaldoImportedPath},
		{BackpackRespaldo2SourcePath, BackpackRespaldo2ImportedPath},
		{BackpackStrapAdjustSourcePath, BackpackStrapAdjustImportedPath},
		{BackpackStrapBackSourcePath, BackpackStrapBackImportedPath},
		{PhonePrimarySourcePath, PhonePrimaryImportedPath},
		{PhonePart01SourcePath, PhonePart01ImportedPath},
		{PhonePart02SourcePath, PhonePart02ImportedPath},
		{UmbrellaPrimarySourcePath, UmbrellaPrimaryImportedPath},
		{UmbrellaPart01SourcePath, UmbrellaPart01ImportedPath},
		{UmbrellaPart02SourcePath, UmbrellaPart02ImportedPath}};
	TArray<UObject*> AssetsToPersist;
	for (const TPair<FString, FString>& MovePair : ImportedAssetMoves)
	{
		UObject* ImportedAsset = EnsureImportedAssetMoved(AssetSubsystem, MovePair.Key, MovePair.Value, OutError);
		if (ImportedAsset == nullptr)
		{
			return false;
		}
		AssetsToPersist.AddUnique(ImportedAsset);
	}

	const TArray<TPair<FString, FString>> AuthoritativeMeshCopies = {
		{ChargerMeshImportedPath, ChargerMeshAuthorityPath},
		{LandingPadMeshImportedPath, LandingPadMeshAuthorityPath},
		{RadioMeshImportedPath, RadioMeshAuthorityPath},
		{PoliceSignMeshImportedPath, PoliceSignMeshAuthorityPath},
		{TrafficLightMeshImportedPath, TrafficLightMeshAuthorityPath},
		{PoliceTapeMeshImportedPath, PoliceTapeMeshAuthorityPath},
		{TrafficConeMeshImportedPath, TrafficConeMeshAuthorityPath},
		{DeliveryBagMeshImportedPath, DeliveryBagMeshAuthorityPath},
		{AmbulanceMeshImportedPath, AmbulanceMeshAuthorityPath}};
	for (const TPair<FString, FString>& CopyPair : AuthoritativeMeshCopies)
	{
		UObject* AuthorityAsset = EnsureAuthoritativeAssetDuplicate(AssetSubsystem, CopyPair.Key, CopyPair.Value, OutError);
		if (AuthorityAsset == nullptr)
		{
			return false;
		}
		AssetsToPersist.AddUnique(AuthorityAsset);
	}

	const TArray<FBootstrapMaterialFixup> MaterialFixups = {
		{TrafficLightMeshAuthorityPath, {TrafficLightMaterialPath}, TEXT("TrafficLight")},
		{TrafficConeMeshAuthorityPath, {TrafficConeMaterialPath}, TEXT("TrafficCone")}};
	if (!ApplyBootstrapMaterialFixups(AssetSubsystem, MaterialFixups, OutError))
	{
		return false;
	}

	UBlueprint* TriggerNoFlyBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, TriggerNoFlyAssetPath, AAeroTriggerZoneBase::StaticClass(), OutError);
	if (TriggerNoFlyBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* TriggerConstructionBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, TriggerConstructionAssetPath, AAeroTriggerZoneBase::StaticClass(), OutError);
	if (TriggerConstructionBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* TriggerGenericBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, TriggerGenericAssetPath, AAeroTriggerZoneBase::StaticClass(), OutError);
	if (TriggerGenericBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* PedBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, PedBlueprintAssetPath, PedParentClass, OutError);
	if (PedBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* VehiclePoliceBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, VehiclePoliceAssetPath, AAeroDynamicVisibleActorBase::StaticClass(), OutError);
	if (VehiclePoliceBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* VehicleAmbulanceBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, VehicleAmbulanceAssetPath, AAeroDynamicVisibleActorBase::StaticClass(), OutError);
	if (VehicleAmbulanceBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* VehicleServiceBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, VehicleServiceAssetPath, VehicleServiceParentClass, OutError);
	if (VehicleServiceBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* PropBackpackBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, PropBackpackAssetPath, AAeroCompositeMeshActorBase::StaticClass(), OutError);
	if (PropBackpackBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* PropPhoneBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, PropPhoneAssetPath, AAeroCompositeMeshActorBase::StaticClass(), OutError);
	if (PropPhoneBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* PropUmbrellaBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, PropUmbrellaAssetPath, AAeroCompositeMeshActorBase::StaticClass(), OutError);
	if (PropUmbrellaBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* PropBarrierBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, PropBarrierAssetPath, AAeroDynamicVisibleActorBase::StaticClass(), OutError);
	if (PropBarrierBlueprint == nullptr)
	{
		return false;
	}

	UBlueprint* UavInspectionBlueprint = CreateOrLoadChildBlueprint(AssetSubsystem, UavInspectionAssetPath, UavInspectionParentClass, OutError);
	if (UavInspectionBlueprint == nullptr)
	{
		return false;
	}

	UPedestrianVariantCatalog* PedCatalog = CreateOrLoadPedestrianVariantCatalog(AssetSubsystem, PedCatalogAssetPath, OutError);
	if (PedCatalog == nullptr)
	{
		return false;
	}

	UCrowdAppearancePool* CrowdAppearancePool = CreateOrLoadCrowdAppearancePool(AssetSubsystem, CrowdAppearancePoolAssetPath, OutError);
	if (CrowdAppearancePool == nullptr)
	{
		return false;
	}

	UCrowdRoleProfile* CrowdRoleProfile = CreateOrLoadCrowdRoleProfile(AssetSubsystem, CrowdRoleProfileAssetPath, OutError);
	if (CrowdRoleProfile == nullptr)
	{
		return false;
	}

	if (!UpdatePedestrianVariantCatalog(PedCatalog, OutError))
	{
		return false;
	}

	if (!UpdateCrowdAppearancePool(CrowdAppearancePool, OutError))
	{
		return false;
	}

	if (!UpdateCrowdRoleProfile(CrowdRoleProfile, OutError))
	{
		return false;
	}

	if (!ConfigurePedestrianBlueprintDefaults(AssetSubsystem, PedBlueprint, PedCatalog, OutError))
	{
		return false;
	}

	if (!ConfigureDynamicVisibleBlueprint(
			AssetSubsystem,
			VehiclePoliceBlueprint,
			{
				PoliceBodyImportedPath,
				PoliceGlassImportedPath,
				PoliceInteriorImportedPath,
				PoliceShadowImportedPath,
				PoliceWheelFlImportedPath,
				PoliceWheelFrImportedPath,
				PoliceWheelRlImportedPath,
				PoliceWheelRrImportedPath
			},
			FVector(120.0f, 60.0f, 60.0f),
			OutError))
	{
		return false;
	}

	if (!ConfigureDynamicVisibleBlueprint(
			AssetSubsystem,
			VehicleAmbulanceBlueprint,
			{
				AmbulanceMeshAuthorityPath
			},
			FVector(145.0f, 68.0f, 78.0f),
			OutError))
	{
		return false;
	}

	if (!ConfigureCompositeMeshBlueprint(
			AssetSubsystem,
			PropBackpackBlueprint,
			{
				BackpackPrimaryImportedPath,
				BackpackExtrasImportedPath,
				BackpackRespaldoImportedPath,
				BackpackRespaldo2ImportedPath,
				BackpackStrapAdjustImportedPath,
				BackpackStrapBackImportedPath
			},
			OutError))
	{
		return false;
	}

	if (!ConfigureCompositeMeshBlueprint(
			AssetSubsystem,
			PropPhoneBlueprint,
			{
				PhonePrimaryImportedPath,
				PhonePart01ImportedPath,
				PhonePart02ImportedPath
			},
			OutError))
	{
		return false;
	}

	if (!ConfigureCompositeMeshBlueprint(
			AssetSubsystem,
			PropUmbrellaBlueprint,
			{
				UmbrellaPrimaryImportedPath,
				UmbrellaPart01ImportedPath,
				UmbrellaPart02ImportedPath
			},
			OutError))
	{
		return false;
	}

	if (!ConfigureDynamicVisibleBlueprint(
			AssetSubsystem,
			PropBarrierBlueprint,
			{
				ConstructionFenceAuthorityPath
			},
			FVector(120.0f, 25.0f, 50.0f),
			OutError))
	{
		return false;
	}

	const TArray<UObject*> BootstrapAssets = {
		TriggerNoFlyBlueprint,
		TriggerConstructionBlueprint,
		TriggerGenericBlueprint,
		PedBlueprint,
		VehiclePoliceBlueprint,
		VehicleAmbulanceBlueprint,
		VehicleServiceBlueprint,
		PropBackpackBlueprint,
		PropPhoneBlueprint,
		PropUmbrellaBlueprint,
		PropBarrierBlueprint,
		UavInspectionBlueprint,
		PedCatalog,
		CrowdAppearancePool,
		CrowdRoleProfile};
	for (UObject* Asset : BootstrapAssets)
	{
		AssetsToPersist.AddUnique(Asset);
	}

	if (!SaveAssets(AssetSubsystem, AssetsToPersist, OutError))
	{
		return false;
	}

	return ValidateAeroWorldContentAssets(OutError);
}

bool UAeroEditorToolsSubsystem::ValidateAeroWorldContentAssets(FString& OutError) const
{
	UEditorAssetSubsystem* AssetSubsystem = ResolveEditorAssetSubsystem(OutError);
	if (AssetSubsystem == nullptr)
	{
		return false;
	}

	TArray<FString> Errors;
	const TArray<FString> ExpectedAssetPaths = {
		TriggerNoFlyAssetPath,
		TriggerConstructionAssetPath,
		TriggerGenericAssetPath,
		PedBlueprintAssetPath,
		VehiclePoliceAssetPath,
		VehicleAmbulanceAssetPath,
		VehicleServiceAssetPath,
		PropBackpackAssetPath,
		PropPhoneAssetPath,
		PropUmbrellaAssetPath,
		PropBarrierAssetPath,
		UavInspectionAssetPath,
		PedCatalogAssetPath,
		CrowdAppearancePoolAssetPath,
		CrowdRoleProfileAssetPath,
		ChargerMeshAuthorityPath,
		LandingPadMeshAuthorityPath,
		RadioMeshAuthorityPath,
		PoliceSignMeshAuthorityPath,
		TrafficLightMeshAuthorityPath,
		PoliceTapeMeshAuthorityPath,
		DeliveryBagMeshAuthorityPath,
		AmbulanceMeshAuthorityPath,
		ConstructionFenceAuthorityPath,
		TrafficConeMeshAuthorityPath};
	for (const FString& AssetPath : ExpectedAssetPaths)
	{
		if (!AssetSubsystem->DoesAssetExist(AssetPath))
		{
			Errors.Add(FString::Printf(TEXT("Expected authoritative asset is missing: %s"), *AssetPath));
		}
	}

	FString ParentLoadError;
	UClass* PedParentClass = LoadParentBlueprintClass(AssetSubsystem, PedBlueprintParentPath, ParentLoadError);
	UClass* VehicleServiceParentClass = LoadParentBlueprintClass(AssetSubsystem, VehicleServiceParentPath, ParentLoadError);
	UClass* UavInspectionParentClass = LoadParentBlueprintClass(AssetSubsystem, UavInspectionParentPath, ParentLoadError);
	if (PedParentClass == nullptr || VehicleServiceParentClass == nullptr || UavInspectionParentClass == nullptr)
	{
		OutError = ParentLoadError;
		return false;
	}

	ValidateBlueprintParent(AssetSubsystem, TriggerNoFlyAssetPath, AAeroTriggerZoneBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, TriggerConstructionAssetPath, AAeroTriggerZoneBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, TriggerGenericAssetPath, AAeroTriggerZoneBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, PedBlueprintAssetPath, PedParentClass, Errors);
	ValidateBlueprintParent(AssetSubsystem, VehiclePoliceAssetPath, AAeroDynamicVisibleActorBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, VehicleAmbulanceAssetPath, AAeroDynamicVisibleActorBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, VehicleServiceAssetPath, VehicleServiceParentClass, Errors);
	ValidateBlueprintParent(AssetSubsystem, PropBackpackAssetPath, AAeroCompositeMeshActorBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, PropPhoneAssetPath, AAeroCompositeMeshActorBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, PropUmbrellaAssetPath, AAeroCompositeMeshActorBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, PropBarrierAssetPath, AAeroDynamicVisibleActorBase::StaticClass(), Errors);
	ValidateBlueprintParent(AssetSubsystem, UavInspectionAssetPath, UavInspectionParentClass, Errors);
	ValidateDynamicVisibleMeshes(AssetSubsystem, VehiclePoliceAssetPath, 8, Errors);
	ValidateDynamicVisibleMeshes(AssetSubsystem, VehicleAmbulanceAssetPath, 1, Errors);
	ValidateDynamicVisibleMeshes(AssetSubsystem, PropBarrierAssetPath, 1, Errors);

	ValidateTriggerInheritance(AssetSubsystem, TriggerNoFlyAssetPath, Errors);
	ValidateTriggerInheritance(AssetSubsystem, TriggerConstructionAssetPath, Errors);
	ValidateTriggerInheritance(AssetSubsystem, TriggerGenericAssetPath, Errors);
	ValidatePedCatalog(AssetSubsystem, Errors);
	ValidateCrowdAppearancePool(AssetSubsystem, Errors);
	ValidateCrowdRoleProfile(AssetSubsystem, Errors);
	ValidatePedestrianRuntimeSettingsDefaults(Errors);
	ValidateAssetCatalogMappings(AssetSubsystem, Errors);

	if (Errors.Num() > 0)
	{
		OutError = FString::Join(Errors, TEXT("\n"));
		return false;
	}

	OutError.Reset();
	return true;
}
