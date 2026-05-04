#include "AeroSemanticTypes.h"

#include "AeroEditorToolsSubsystem.h"
#include "CrowdAppearancePool.h"
#include "CrowdRoleProfile.h"
#include "Dom/JsonObject.h"
#include "Editor.h"
#include "Misc/AutomationTest.h"
#include "PedestrianRuntimeSettings.h"
#include "PedestrianVariantCatalog.h"
#include "Subsystems/EditorAssetSubsystem.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAeroSemanticParsingTest,
	"Aero.Editor.SemanticRuntime.ParseEnums",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FAeroSemanticParsingTest::RunTest(const FString& Parameters)
{
	TestEqual(TEXT("hit parsing"), AeroParseFeedbackMode(TEXT("hit")), EAeroFeedbackMode::Hit);
	TestEqual(TEXT("overlap parsing"), AeroParseFeedbackMode(TEXT("overlap")), EAeroFeedbackMode::Overlap);
	TestEqual(TEXT("both parsing"), AeroParseFeedbackMode(TEXT("both")), EAeroFeedbackMode::Both);
	TestEqual(TEXT("box parsing"), AeroParseTriggerShapeKind(TEXT("box_volume")), EAeroTriggerShapeKind::Box);
	TestEqual(TEXT("sphere parsing"), AeroParseTriggerShapeKind(TEXT("sphere_volume")), EAeroTriggerShapeKind::Sphere);
	TestEqual(TEXT("polygon parsing"), AeroParseTriggerShapeKind(TEXT("polygon_prism")), EAeroTriggerShapeKind::PolygonPrism);
	TestEqual(TEXT("teleport movement parsing"), AeroParseMovementMode(TEXT("teleport")), EAeroMovementMode::Teleport);
	TestEqual(TEXT("sweep follow movement parsing"), AeroParseMovementMode(TEXT("sweep_follow")), EAeroMovementMode::SweepFollow);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAeroFeedbackJsonTest,
	"Aero.Editor.SemanticRuntime.FeedbackJson",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FAeroFeedbackJsonTest::RunTest(const FString& Parameters)
{
	FAeroFeedbackEvent Event;
	Event.Type = TEXT("collision");
	Event.EventId = TEXT("evt_001");
	Event.Tick = 12;
	Event.FrameId = 24;
	Event.EpisodeId = TEXT("ep_001");
	Event.SourceEntityId = TEXT("uav_01");
	Event.OtherEntityId = TEXT("barrier_01");
	Event.SourceActorId = TEXT("BP_FlyingPawn_1");
	Event.OtherActorId = TEXT("Barrier_1");
	Event.SourceLogicalAssetId = TEXT("uav.inspect.quad.v1");
	Event.OtherLogicalAssetId = TEXT("facility.barrier.basic");
	Event.SourceTags = {TEXT("uav")};
	Event.OtherTags = {TEXT("barrier")};
	Event.Collision.ContactPointEnuM = FVector(1.0, 2.0, 3.0);
	Event.Collision.ContactNormalEnu = FVector(0.0, -1.0, 0.0);
	Event.Collision.RelativeSpeedMps = 8.6;
	Event.Collision.Impulse = 320.0;
	Event.Collision.bBlocking = true;

	const TSharedPtr<FJsonObject> JsonObject = AeroFeedbackEventToJson(Event);
	TestTrue(TEXT("feedback json created"), JsonObject.IsValid());
	if (!JsonObject.IsValid())
	{
		return false;
	}

	TestEqual(TEXT("type preserved"), JsonObject->GetStringField(TEXT("type")), FString(TEXT("collision")));
	TestEqual(TEXT("source entity preserved"), JsonObject->GetStringField(TEXT("source_entity_id")), FString(TEXT("uav_01")));
	TestEqual(TEXT("other entity preserved"), JsonObject->GetStringField(TEXT("other_entity_id")), FString(TEXT("barrier_01")));
	TestEqual(TEXT("tick preserved"), static_cast<int32>(JsonObject->GetNumberField(TEXT("tick"))), 12);
	TestEqual(TEXT("frame preserved"), static_cast<int32>(JsonObject->GetNumberField(TEXT("frame_id"))), 24);
	TestEqual(TEXT("blocking preserved"), JsonObject->GetBoolField(TEXT("blocking")), true);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAeroVisualStateJsonTest,
	"Aero.Editor.SemanticRuntime.VisualStateJson",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FAeroVisualStateJsonTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FJsonObject> JsonObject = MakeShared<FJsonObject>();
	JsonObject->SetStringField(TEXT("mode"), TEXT("observe"));
	JsonObject->SetStringField(TEXT("variant_id"), TEXT("adult_male_commuter"));
	JsonObject->SetStringField(TEXT("montage_tag"), TEXT("observe"));
	JsonObject->SetBoolField(TEXT("lights_on"), true);
	JsonObject->SetStringField(TEXT("material_variant"), TEXT("emergency"));

	FAeroVisualState VisualState;
	TestTrue(TEXT("visual state parsed"), AeroVisualStateFromJson(JsonObject, VisualState));
	TestEqual(TEXT("visual mode preserved"), VisualState.Mode, FString(TEXT("observe")));
	TestEqual(TEXT("visual variant preserved"), VisualState.VariantId, FName(TEXT("adult_male_commuter")));
	TestEqual(TEXT("visual montage preserved"), VisualState.MontageTag, FName(TEXT("observe")));
	TestTrue(TEXT("visual lights flag preserved"), VisualState.bHasLightsOn && VisualState.bLightsOn);
	TestEqual(TEXT("visual material preserved"), VisualState.MaterialVariant, FString(TEXT("emergency")));

	const TSharedPtr<FJsonObject> RoundTripJson = AeroVisualStateToJson(VisualState);
	TestTrue(TEXT("visual state roundtrip json created"), RoundTripJson.IsValid());
	if (RoundTripJson.IsValid())
	{
		TestEqual(TEXT("roundtrip mode preserved"), RoundTripJson->GetStringField(TEXT("mode")), FString(TEXT("observe")));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAeroWorldContentBootstrapTest,
	"Aero.Editor.WorldContent.BootstrapAndValidate",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FAeroWorldContentBootstrapTest::RunTest(const FString& Parameters)
{
	if (GEditor == nullptr)
	{
		AddError(TEXT("GEditor is unavailable."));
		return false;
	}

	UAeroEditorToolsSubsystem* Tools = GEditor->GetEditorSubsystem<UAeroEditorToolsSubsystem>();
	UEditorAssetSubsystem* AssetSubsystem = GEditor->GetEditorSubsystem<UEditorAssetSubsystem>();
	TestNotNull(TEXT("AeroEditorToolsSubsystem available"), Tools);
	TestNotNull(TEXT("EditorAssetSubsystem available"), AssetSubsystem);
	if (Tools == nullptr || AssetSubsystem == nullptr)
	{
		return false;
	}

	FString Error;
	TestTrue(TEXT("bootstrap succeeds"), Tools->BootstrapAeroWorldContentAssets(Error));
	if (!Error.IsEmpty())
	{
		AddInfo(Error);
	}

	Error.Reset();
	TestTrue(TEXT("bootstrap is idempotent"), Tools->BootstrapAeroWorldContentAssets(Error));
	if (!Error.IsEmpty())
	{
		AddInfo(Error);
	}

	Error.Reset();
	TestTrue(TEXT("validation succeeds"), Tools->ValidateAeroWorldContentAssets(Error));
	if (!Error.IsEmpty())
	{
		AddInfo(Error);
	}

	const TArray<FString> AssetPaths = {
		TEXT("/AeroWorldContent/Blueprints/Triggers/BP_AW_Trigger_NoFly_Box_01.BP_AW_Trigger_NoFly_Box_01"),
		TEXT("/AeroWorldContent/Blueprints/Triggers/BP_AW_Trigger_Hazard_Construction_Box_01.BP_AW_Trigger_Hazard_Construction_Box_01"),
		TEXT("/AeroWorldContent/Blueprints/Triggers/BP_AW_Trigger_Hazard_Generic_Box_01.BP_AW_Trigger_Hazard_Generic_Box_01"),
		TEXT("/AeroWorldContent/Blueprints/Pedestrians/BP_AW_Pedestrian_CityOps_01.BP_AW_Pedestrian_CityOps_01"),
		TEXT("/AeroWorldContent/Blueprints/Vehicles/BP_AW_Vehicle_Emergency_SUV_01.BP_AW_Vehicle_Emergency_SUV_01"),
		TEXT("/AeroWorldContent/Blueprints/Vehicles/BP_AW_Vehicle_Emergency_Ambulance_01.BP_AW_Vehicle_Emergency_Ambulance_01"),
		TEXT("/AeroWorldContent/Blueprints/Vehicles/BP_AW_Vehicle_Service_Box_01.BP_AW_Vehicle_Service_Box_01"),
		TEXT("/AeroWorldContent/Blueprints/Props/BP_AW_Prop_Service_Backpack_01.BP_AW_Prop_Service_Backpack_01"),
		TEXT("/AeroWorldContent/Blueprints/Props/BP_AW_Prop_Misc_Phone_01.BP_AW_Prop_Misc_Phone_01"),
		TEXT("/AeroWorldContent/Blueprints/Props/BP_AW_Prop_Misc_Umbrella_01.BP_AW_Prop_Misc_Umbrella_01"),
		TEXT("/AeroWorldContent/Blueprints/Props/BP_AW_Prop_Roadwork_Barrier_01.BP_AW_Prop_Roadwork_Barrier_01"),
		TEXT("/AeroWorldContent/Blueprints/UAV/BP_AW_UAV_Inspection_Quad_01.BP_AW_UAV_Inspection_Quad_01"),
		TEXT("/AeroWorldContent/DataAssets/Ped/DA_AW_PedVariants_CityOps_01.DA_AW_PedVariants_CityOps_01"),
		TEXT("/AeroWorldContent/DataAssets/Crowd/DA_AW_CrowdAppearancePool_CityOps_01.DA_AW_CrowdAppearancePool_CityOps_01"),
		TEXT("/AeroWorldContent/DataAssets/Crowd/DA_AW_CrowdRoleProfile_CityOps_Default_01.DA_AW_CrowdRoleProfile_CityOps_Default_01"),
		TEXT("/AeroWorldContent/Meshes/Facilities/Charger/SM_AW_Facility_Charger_01.SM_AW_Facility_Charger_01"),
		TEXT("/AeroWorldContent/Meshes/Facilities/LandingPad/SM_AW_Facility_LandingPad_01.SM_AW_Facility_LandingPad_01"),
		TEXT("/AeroWorldContent/Meshes/Infrastructure/Radio/SM_AW_Facility_RadioTower_01.SM_AW_Facility_RadioTower_01"),
		TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/SM_AW_Prop_TrafficControl_PoliceSign_01.SM_AW_Prop_TrafficControl_PoliceSign_01"),
		TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/SM_AW_Prop_TrafficControl_TrafficLight_01.SM_AW_Prop_TrafficControl_TrafficLight_01"),
		TEXT("/AeroWorldContent/Meshes/Infrastructure/TrafficControl/SM_AW_Prop_Incident_PoliceTape_01.SM_AW_Prop_Incident_PoliceTape_01"),
		TEXT("/AeroWorldContent/Meshes/Props/Roadwork/SM_AW_Prop_ConstructionFence_01.SM_AW_Prop_ConstructionFence_01"),
		TEXT("/AeroWorldContent/Meshes/Props/Service/SM_AW_Prop_Service_DeliveryBag_01.SM_AW_Prop_Service_DeliveryBag_01"),
		TEXT("/AeroWorldContent/Meshes/Vehicles/Emergency/SM_AW_Vehicle_Emergency_Ambulance_01.SM_AW_Vehicle_Emergency_Ambulance_01")};
	for (const FString& AssetPath : AssetPaths)
	{
		TestTrue(FString::Printf(TEXT("asset exists: %s"), *AssetPath), AssetSubsystem->DoesAssetExist(AssetPath));
	}

	UPedestrianVariantCatalog* Catalog = Cast<UPedestrianVariantCatalog>(
		AssetSubsystem->LoadAsset(TEXT("/AeroWorldContent/DataAssets/Ped/DA_AW_PedVariants_CityOps_01.DA_AW_PedVariants_CityOps_01")));
	TestNotNull(TEXT("pedestrian variant catalog exists"), Catalog);
	if (Catalog != nullptr)
	{
		const TArray<FName> ExpectedIds = {
			FName(TEXT("adult_male_commuter")),
			FName(TEXT("adult_female_commuter")),
			FName(TEXT("child_crossing")),
			FName(TEXT("elder_observer"))};
		for (const FName VariantId : ExpectedIds)
		{
			FPedVariantSpec VariantSpec;
			TestTrue(
				FString::Printf(TEXT("catalog contains variant: %s"), *VariantId.ToString()),
				Catalog->FindVariantById(VariantId, VariantSpec));
			TestEqual(FString::Printf(TEXT("ground offset default for %s"), *VariantId.ToString()), static_cast<int32>(VariantSpec.GroundContactOffsetCm), 0);
		}
	}

	UCrowdAppearancePool* AppearancePool = Cast<UCrowdAppearancePool>(
		AssetSubsystem->LoadAsset(TEXT("/AeroWorldContent/DataAssets/Crowd/DA_AW_CrowdAppearancePool_CityOps_01.DA_AW_CrowdAppearancePool_CityOps_01")));
	TestNotNull(TEXT("crowd appearance pool exists"), AppearancePool);
	if (AppearancePool != nullptr)
	{
		TestEqual(TEXT("crowd appearance pool entry count"), AppearancePool->Entries.Num(), 4);

		const FCrowdAppearanceEntry* FemaleAppearance = AppearancePool->Entries.FindByPredicate([](const FCrowdAppearanceEntry& Entry) {
			return Entry.AppearanceId == FName(TEXT("adult_female_commuter"));
		});
		TestNotNull(TEXT("adult female appearance exists"), FemaleAppearance);
		if (FemaleAppearance != nullptr)
		{
			int32 UmbrellaPartCount = 0;
			for (const FCrowdAccessorySpec& Accessory : FemaleAppearance->OptionalAccessories)
			{
				if (Accessory.AccessoryTag == FName(TEXT("umbrella")))
				{
					++UmbrellaPartCount;
					TestEqual(TEXT("female umbrella scale x"), static_cast<int32>(Accessory.RelativeScale.X * 100.0f), 50);
					TestEqual(TEXT("female umbrella scale y"), static_cast<int32>(Accessory.RelativeScale.Y * 100.0f), 50);
					TestEqual(TEXT("female umbrella scale z"), static_cast<int32>(Accessory.RelativeScale.Z * 100.0f), 50);
					TestTrue(TEXT("female umbrella lifted above hand socket"), Accessory.RelativeLocation.Z > 0.0f);
				}
			}
			TestEqual(TEXT("female umbrella uses three grouped meshes"), UmbrellaPartCount, 3);
		}

		const FCrowdAppearanceEntry* ElderAppearance = AppearancePool->Entries.FindByPredicate([](const FCrowdAppearanceEntry& Entry) {
			return Entry.AppearanceId == FName(TEXT("elder_observer"));
		});
		TestNotNull(TEXT("elder appearance exists"), ElderAppearance);
		if (ElderAppearance != nullptr)
		{
			int32 UmbrellaPartCount = 0;
			for (const FCrowdAccessorySpec& Accessory : ElderAppearance->OptionalAccessories)
			{
				if (Accessory.AccessoryTag == FName(TEXT("umbrella")))
				{
					++UmbrellaPartCount;
					TestEqual(TEXT("elder umbrella scale x"), static_cast<int32>(Accessory.RelativeScale.X * 100.0f), 50);
					TestTrue(TEXT("elder umbrella lifted above hand socket"), Accessory.RelativeLocation.Z > 0.0f);
				}
			}
			TestEqual(TEXT("elder umbrella uses three grouped meshes"), UmbrellaPartCount, 3);
		}
	}

	UCrowdRoleProfile* RoleProfile = Cast<UCrowdRoleProfile>(
		AssetSubsystem->LoadAsset(TEXT("/AeroWorldContent/DataAssets/Crowd/DA_AW_CrowdRoleProfile_CityOps_Default_01.DA_AW_CrowdRoleProfile_CityOps_Default_01")));
	TestNotNull(TEXT("crowd role profile exists"), RoleProfile);
	if (RoleProfile != nullptr)
	{
		TestEqual(TEXT("crowd role count override default"), RoleProfile->CountOverride, -1);
		TestEqual(TEXT("crowd role behavior default"), RoleProfile->DefaultBehaviorMode, FName(TEXT("idle")));
		TestEqual(TEXT("crowd role radius default"), static_cast<int32>(RoleProfile->DefaultSpawnRadius), 1200);
		TestEqual(TEXT("crowd role spacing default"), static_cast<int32>(RoleProfile->DefaultMinSpacing), 120);
		TestEqual(TEXT("crowd role blocked tags cleared"), RoleProfile->BlockedTags.Num(), 0);
	}

	const UPedestrianRuntimeSettings* Settings = GetDefault<UPedestrianRuntimeSettings>();
	TestNotNull(TEXT("pedestrian runtime settings exist"), Settings);
	if (Settings != nullptr)
	{
		TestNotNull(TEXT("default pedestrian class resolves"), Settings->DefaultPedestrianClass.LoadSynchronous());
		TestEqual(TEXT("default spawn variant id"), Settings->DefaultSpawnVariantId, FName(TEXT("adult_female_commuter")));
		TestNotNull(TEXT("default crowd appearance pool resolves"), Settings->DefaultCrowdAppearancePool.LoadSynchronous());
		TestNotNull(TEXT("default crowd role profile resolves"), Settings->DefaultCrowdRoleProfile.LoadSynchronous());
	}

	return true;
}
