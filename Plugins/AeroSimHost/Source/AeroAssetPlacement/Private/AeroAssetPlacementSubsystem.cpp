#include "AeroAssetPlacementSubsystem.h"

#include "AeroFeedbackSubsystem.h"
#include "AeroSemanticRuntimeHelpers.h"
#include "AeroTriggerZoneBase.h"
#include "Components/PrimitiveComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/Blueprint.h"
#include "GameFramework/Actor.h"
#include "GroundPlacementUtils.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "UObject/SoftObjectPath.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
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

TArray<FString> ReadStringArray(const TSharedPtr<FJsonObject>& Object, const FString& FieldName)
{
	TArray<FString> Result;
	const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
	if (!Object.IsValid() || !Object->TryGetArrayField(FieldName, Values) || Values == nullptr)
	{
		return Result;
	}

	for (const TSharedPtr<FJsonValue>& Value : *Values)
	{
		FString Item;
		if (Value.IsValid() && Value->TryGetString(Item))
		{
			Result.Add(Item);
		}
	}
	return Result;
}

void WriteVectorArrayField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FVector& VectorValue)
{
	TArray<TSharedPtr<FJsonValue>> Values;
	Values.Add(MakeShared<FJsonValueNumber>(VectorValue.X));
	Values.Add(MakeShared<FJsonValueNumber>(VectorValue.Y));
	Values.Add(MakeShared<FJsonValueNumber>(VectorValue.Z));
	Object->SetArrayField(FieldName, Values);
}

void WriteRotatorField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FRotator& Rotation)
{
	TSharedPtr<FJsonObject> RotationObject = MakeShared<FJsonObject>();
	RotationObject->SetNumberField(TEXT("roll_deg"), Rotation.Roll);
	RotationObject->SetNumberField(TEXT("pitch_deg"), Rotation.Pitch);
	RotationObject->SetNumberField(TEXT("yaw_deg"), Rotation.Yaw);
	Object->SetObjectField(FieldName, RotationObject);
}

void WriteVectorObjectField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FVector& VectorValue)
{
	TSharedPtr<FJsonObject> VectorObject = MakeShared<FJsonObject>();
	VectorObject->SetNumberField(TEXT("x"), VectorValue.X);
	VectorObject->SetNumberField(TEXT("y"), VectorValue.Y);
	VectorObject->SetNumberField(TEXT("z"), VectorValue.Z);
	Object->SetObjectField(FieldName, VectorObject);
}
}

bool UAeroAssetPlacementSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

void UAeroAssetPlacementSubsystem::SetMapContext(const FString& MapId, const TSharedPtr<FJsonObject>& MapContext)
{
	CurrentMapId = MapId;
	CurrentWorldOriginCm = FVector::ZeroVector;
	if (MapContext.IsValid())
	{
		TryReadVectorField(MapContext, TEXT("world_origin_cm"), CurrentWorldOriginCm);
	}

	if (UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>() : nullptr)
	{
		FeedbackSubsystem->SetWorldOriginCm(CurrentWorldOriginCm);
	}
}

bool UAeroAssetPlacementSubsystem::LoadAssetCatalog(const FString& CatalogPath, FString& OutError)
{
	TSharedPtr<FJsonObject> RootObject;
	if (!LoadJsonObjectFromFile(CatalogPath, RootObject, OutError))
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Assets = nullptr;
	if (!RootObject->TryGetArrayField(TEXT("assets"), Assets) || Assets == nullptr)
	{
		OutError = FString::Printf(TEXT("asset_catalog missing 'assets' array: %s"), *CatalogPath);
		return false;
	}

	TMap<FString, FAeroAssetTemplateDefinition> ParsedTemplates;
	for (const TSharedPtr<FJsonValue>& AssetValue : *Assets)
	{
		const TSharedPtr<FJsonObject> AssetObject = AssetValue.IsValid() ? AssetValue->AsObject() : nullptr;
		if (!AssetObject.IsValid())
		{
			OutError = TEXT("asset_catalog contains a non-object entry.");
			return false;
		}

		FAeroAssetTemplateDefinition TemplateDef;
		if (!ParseTemplateDefinition(AssetObject, TemplateDef, OutError))
		{
			return false;
		}

		if (TemplateDef.SpawnBackend.Equals(TEXT("semantic_only"), ESearchCase::IgnoreCase) && TemplateDef.bRenderRequired)
		{
			OutError = FString::Printf(
				TEXT("asset_catalog entry '%s' is invalid: semantic_only assets cannot have render_required=true."),
				*TemplateDef.LogicalAssetId);
			return false;
		}
		if (!TemplateDef.SpawnBackend.Equals(TEXT("semantic_only"), ESearchCase::IgnoreCase) &&
			!TemplateDef.SpawnBackend.Equals(TEXT("trigger_zone"), ESearchCase::IgnoreCase) &&
			TemplateDef.bRenderRequired && TemplateDef.UEAssetPath.IsEmpty())
		{
			OutError = FString::Printf(
				TEXT("asset_catalog entry '%s' is invalid: render_required=true but ue_asset_path is empty."),
				*TemplateDef.LogicalAssetId);
			return false;
		}

		ParsedTemplates.Add(TemplateDef.LogicalAssetId, TemplateDef);
	}

	TemplatesById = MoveTemp(ParsedTemplates);
	return true;
}

bool UAeroAssetPlacementSubsystem::LoadScenarioObjects(const FString& ScenarioPath, FString& OutError)
{
	TSharedPtr<FJsonObject> RootObject;
	if (!LoadJsonObjectFromFile(ScenarioPath, RootObject, OutError))
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Objects = nullptr;
	if (!RootObject->TryGetArrayField(TEXT("objects"), Objects) || Objects == nullptr)
	{
		if (!RootObject->TryGetArrayField(TEXT("entities"), Objects) || Objects == nullptr)
		{
			OutError = FString::Printf(TEXT("scenario_objects missing 'objects' or scene_setup 'entities' array: %s"), *ScenarioPath);
			return false;
		}
	}

	for (TPair<FString, FAeroAssetInstanceState>& ExistingPair : InstancesById)
	{
		if (!ExistingPair.Value.bDynamic)
		{
			DestroyActorForInstance(ExistingPair.Value);
		}
	}

	TMap<FString, FAeroAssetInstanceState> NextInstances;
	for (const TPair<FString, FAeroAssetInstanceState>& Pair : InstancesById)
	{
		if (Pair.Value.bDynamic)
		{
			NextInstances.Add(Pair.Key, Pair.Value);
		}
	}

	for (const TSharedPtr<FJsonValue>& ObjectValue : *Objects)
	{
		const TSharedPtr<FJsonObject> Object = ObjectValue.IsValid() ? ObjectValue->AsObject() : nullptr;
		if (!Object.IsValid())
		{
			OutError = TEXT("scenario_objects contains a non-object entry.");
			return false;
		}

		FAeroAssetInstanceState Instance;
		if (!ParseScenarioObject(Object, Instance, OutError))
		{
			return false;
		}

		NextInstances.Add(Instance.InstanceId, Instance);
	}

	InstancesById = MoveTemp(NextInstances);
	return SpawnScenarioActors(OutError);
}

const FAeroAssetTemplateDefinition* UAeroAssetPlacementSubsystem::FindTemplate(const FString& LogicalAssetId) const
{
	return TemplatesById.Find(LogicalAssetId);
}

const FAeroAssetInstanceState* UAeroAssetPlacementSubsystem::FindInstance(const FString& InstanceId) const
{
	return InstancesById.Find(InstanceId);
}

TSharedPtr<FJsonObject> UAeroAssetPlacementSubsystem::SpawnAsset(const TSharedPtr<FJsonObject>& Payload, FString& OutError)
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("SpawnAsset payload is invalid.");
		return nullptr;
	}

	FString TemplateId;
	if (!Payload->TryGetStringField(TEXT("template_id"), TemplateId) && !Payload->TryGetStringField(TEXT("logical_asset_id"), TemplateId))
	{
		OutError = TEXT("SpawnAsset requires template_id or logical_asset_id.");
		return nullptr;
	}

	FVector PositionEnuM = FVector::ZeroVector;
	FVector PositionWorldCm = FVector::ZeroVector;
	FRotator RotationDeg = FRotator::ZeroRotator;
	if (!TryResolvePayloadPose(Payload, PositionEnuM, &PositionWorldCm, RotationDeg))
	{
		OutError = TEXT("SpawnAsset requires pose_enu_m/pose_world_cm or position_enu_m/position_world_cm.");
		return nullptr;
	}

	const TArray<FString> QueryTags = ReadStringArray(Payload, TEXT("tags"));
	FAeroVisualState VisualState;
	const bool bHasVisualState = TryReadVisualStateField(Payload, TEXT("visual_state"), VisualState);
	FString EntityId;
	Payload->TryGetStringField(TEXT("entity_id"), EntityId);
	FString InstanceId;
	if (!Payload->TryGetStringField(TEXT("asset_id"), InstanceId))
	{
		InstanceId = FString::Printf(TEXT("dyn_asset_%s"), *FGuid::NewGuid().ToString(EGuidFormats::Digits));
	}

	if (!SpawnOrUpdateProxy(InstanceId, TemplateId, PositionEnuM, RotationDeg, QueryTags, EntityId, bHasVisualState ? &VisualState : nullptr, OutError))
	{
		return nullptr;
	}

	const FAeroAssetInstanceState* ResolvedInstance = FindInstance(InstanceId);
	const FVector ResolvedEnuM = ResolvedInstance != nullptr ? ResolvedInstance->PositionEnuM : PositionEnuM;
	const FVector ResolvedWorldCm = ResolvedInstance != nullptr ? ResolvedInstance->LastResolvedWorldLocationCm : ConvertEnuMetersToWorldCm(ResolvedEnuM);
	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("asset_id"), InstanceId);
	Result->SetStringField(TEXT("logical_asset_id"), TemplateId);
	WriteVectorArrayField(Result, TEXT("position_enu_m"), ResolvedEnuM);
	WriteVectorObjectField(Result, TEXT("position_world_cm"), ResolvedWorldCm);
	WriteRotatorField(Result, TEXT("rotation_deg"), RotationDeg);
	if (ResolvedInstance != nullptr)
	{
		if (ResolvedInstance->Actor.IsValid())
		{
			Result->SetStringField(TEXT("actor_name"), ResolvedInstance->Actor->GetName());
		}
		if (!ResolvedInstance->LastGroundSource.IsEmpty())
		{
			Result->SetStringField(TEXT("ground_source"), ResolvedInstance->LastGroundSource);
		}
	}
	return Result;
}

TSharedPtr<FJsonObject> UAeroAssetPlacementSubsystem::MoveAsset(const TSharedPtr<FJsonObject>& Payload, FString& OutError)
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("MoveAsset payload is invalid.");
		return nullptr;
	}

	FString InstanceId;
	if (!Payload->TryGetStringField(TEXT("asset_id"), InstanceId) && !Payload->TryGetStringField(TEXT("instance_id"), InstanceId))
	{
		OutError = TEXT("MoveAsset requires asset_id.");
		return nullptr;
	}

	FAeroAssetInstanceState* Instance = InstancesById.Find(InstanceId);
	if (Instance == nullptr)
	{
		OutError = FString::Printf(TEXT("Unknown asset_id: %s"), *InstanceId);
		return nullptr;
	}

	FVector PositionEnuM = Instance->PositionEnuM;
	FVector PositionWorldCm = ConvertEnuMetersToWorldCm(PositionEnuM);
	FRotator RotationDeg = Instance->RotationDeg;
	FAeroVisualState VisualState;
	const bool bHasVisualState = TryReadVisualStateField(Payload, TEXT("visual_state"), VisualState);
	if (!TryResolvePayloadPose(Payload, PositionEnuM, &PositionWorldCm, RotationDeg))
	{
		Payload->TryGetNumberField(TEXT("yaw_deg"), RotationDeg.Yaw);
	}

	if (!SpawnOrUpdateProxy(
			InstanceId,
			Instance->LogicalAssetId,
			PositionEnuM,
			RotationDeg,
			Instance->QueryTags,
			Instance->EntityId,
			bHasVisualState ? &VisualState : nullptr,
			OutError))
	{
		return nullptr;
	}

	const FAeroAssetInstanceState* ResolvedInstance = FindInstance(InstanceId);
	const FVector ResolvedEnuM = ResolvedInstance != nullptr ? ResolvedInstance->PositionEnuM : PositionEnuM;
	const FVector ResolvedWorldCm = ResolvedInstance != nullptr ? ResolvedInstance->LastResolvedWorldLocationCm : ConvertEnuMetersToWorldCm(ResolvedEnuM);
	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("asset_id"), InstanceId);
	WriteVectorArrayField(Result, TEXT("position_enu_m"), ResolvedEnuM);
	WriteVectorObjectField(Result, TEXT("position_world_cm"), ResolvedWorldCm);
	WriteRotatorField(Result, TEXT("rotation_deg"), RotationDeg);
	if (ResolvedInstance != nullptr)
	{
		if (ResolvedInstance->Actor.IsValid())
		{
			Result->SetStringField(TEXT("actor_name"), ResolvedInstance->Actor->GetName());
		}
		if (!ResolvedInstance->LastGroundSource.IsEmpty())
		{
			Result->SetStringField(TEXT("ground_source"), ResolvedInstance->LastGroundSource);
		}
	}
	return Result;
}

TSharedPtr<FJsonObject> UAeroAssetPlacementSubsystem::RemoveAsset(const TSharedPtr<FJsonObject>& Payload, FString& OutError)
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("RemoveAsset payload is invalid.");
		return nullptr;
	}

	FString InstanceId;
	if (!Payload->TryGetStringField(TEXT("asset_id"), InstanceId) && !Payload->TryGetStringField(TEXT("instance_id"), InstanceId))
	{
		OutError = TEXT("RemoveAsset requires asset_id.");
		return nullptr;
	}

	if (!RemoveProxy(InstanceId, OutError))
	{
		return nullptr;
	}

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("asset_id"), InstanceId);
	Result->SetBoolField(TEXT("removed"), true);
	return Result;
}

TSharedPtr<FJsonObject> UAeroAssetPlacementSubsystem::ReserveOccupancy(const TSharedPtr<FJsonObject>& Payload, FString& OutError)
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("ReserveOccupancy payload is invalid.");
		return nullptr;
	}

	FString InstanceId;
	FString EntityId;
	if (!Payload->TryGetStringField(TEXT("asset_id"), InstanceId))
	{
		OutError = TEXT("ReserveOccupancy requires asset_id.");
		return nullptr;
	}
	if (!Payload->TryGetStringField(TEXT("entity_id"), EntityId))
	{
		OutError = TEXT("ReserveOccupancy requires entity_id.");
		return nullptr;
	}

	FAeroAssetInstanceState* Instance = InstancesById.Find(InstanceId);
	if (Instance == nullptr)
	{
		OutError = FString::Printf(TEXT("Unknown asset_id: %s"), *InstanceId);
		return nullptr;
	}

	Instance->bReserved = true;
	Instance->ReservedBy = EntityId;

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("asset_id"), InstanceId);
	Result->SetStringField(TEXT("entity_id"), EntityId);
	Result->SetBoolField(TEXT("reserved"), true);
	return Result;
}

TSharedPtr<FJsonObject> UAeroAssetPlacementSubsystem::ReleaseOccupancy(const TSharedPtr<FJsonObject>& Payload, FString& OutError)
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("ReleaseOccupancy payload is invalid.");
		return nullptr;
	}

	FString InstanceId;
	if (!Payload->TryGetStringField(TEXT("asset_id"), InstanceId))
	{
		OutError = TEXT("ReleaseOccupancy requires asset_id.");
		return nullptr;
	}

	FAeroAssetInstanceState* Instance = InstancesById.Find(InstanceId);
	if (Instance == nullptr)
	{
		OutError = FString::Printf(TEXT("Unknown asset_id: %s"), *InstanceId);
		return nullptr;
	}

	Instance->bReserved = false;
	Instance->ReservedBy.Reset();

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("asset_id"), InstanceId);
	Result->SetBoolField(TEXT("reserved"), false);
	return Result;
}

TSharedPtr<FJsonObject> UAeroAssetPlacementSubsystem::QueryNearest(const TSharedPtr<FJsonObject>& Payload, FString& OutError) const
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("QueryNearest payload is invalid.");
		return nullptr;
	}

	FString QueryTag;
	if (!Payload->TryGetStringField(TEXT("tag"), QueryTag))
	{
		OutError = TEXT("QueryNearest requires tag.");
		return nullptr;
	}

	FVector PoseEnuM = FVector::ZeroVector;
	TryReadVectorField(Payload, TEXT("pose_enu_m"), PoseEnuM);

	double RadiusM = 100.0;
	Payload->TryGetNumberField(TEXT("radius_m"), RadiusM);

	const FAeroAssetInstanceState* BestInstance = nullptr;
	double BestDistance = TNumericLimits<double>::Max();
	for (const TPair<FString, FAeroAssetInstanceState>& Pair : InstancesById)
	{
		const FAeroAssetInstanceState& Instance = Pair.Value;
		if (!Instance.bEnabled)
		{
			continue;
		}

		bool bHasTag = false;
		for (const FString& Tag : Instance.QueryTags)
		{
			if (Tag.Equals(QueryTag, ESearchCase::IgnoreCase))
			{
				bHasTag = true;
				break;
			}
		}

		if (!bHasTag)
		{
			continue;
		}

		const double DistanceM = FVector::Distance(PoseEnuM, Instance.PositionEnuM);
		if (DistanceM <= RadiusM && DistanceM < BestDistance)
		{
			BestDistance = DistanceM;
			BestInstance = &Instance;
		}
	}

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("tag"), QueryTag);
	Result->SetBoolField(TEXT("found"), BestInstance != nullptr);
	if (BestInstance != nullptr)
	{
		Result->SetStringField(TEXT("instance_id"), BestInstance->InstanceId);
		Result->SetStringField(TEXT("logical_asset_id"), BestInstance->LogicalAssetId);
		Result->SetNumberField(TEXT("distance_m"), BestDistance);
		WriteVectorArrayField(Result, TEXT("position_enu_m"), BestInstance->PositionEnuM);
		WriteRotatorField(Result, TEXT("rotation_deg"), BestInstance->RotationDeg);
		Result->SetBoolField(TEXT("reserved"), BestInstance->bReserved);
		Result->SetStringField(TEXT("reserved_by"), BestInstance->ReservedBy);
	}
	return Result;
}

bool UAeroAssetPlacementSubsystem::SpawnOrUpdateProxy(
	const FString& InstanceId,
	const FString& LogicalAssetId,
	const FVector& PositionEnuM,
	const FRotator& RotationDeg,
	const TArray<FString>& QueryTags,
	const FString& EntityId,
	const FAeroVisualState* VisualState,
	FString& OutError)
{
	const FAeroAssetTemplateDefinition* TemplateDef = TemplatesById.Find(LogicalAssetId);
	if (TemplateDef == nullptr)
	{
		OutError = FString::Printf(TEXT("Unknown logical_asset_id '%s' for instance '%s'."), *LogicalAssetId, *InstanceId);
		return false;
	}

	FAeroAssetInstanceState* Existing = InstancesById.Find(InstanceId);
	const bool bCreatedInstance = Existing == nullptr;
	if (Existing == nullptr)
	{
		FAeroAssetInstanceState NewInstance;
		NewInstance.InstanceId = InstanceId;
		NewInstance.LogicalAssetId = LogicalAssetId;
		NewInstance.PositionEnuM = PositionEnuM;
		NewInstance.RotationDeg = RotationDeg;
		NewInstance.QueryTags = QueryTags;
		NewInstance.EntityId = EntityId;
		NewInstance.bDynamic = true;
		NewInstance.bEnabled = true;
		NewInstance.MovementMode = TemplateDef->MovementMode;
		if (VisualState != nullptr)
		{
			NewInstance.VisualState = *VisualState;
			NewInstance.bHasVisualState = !VisualState->IsEmpty();
			NewInstance.bVisualStateExplicit = true;
		}
		InstancesById.Add(InstanceId, NewInstance);
		Existing = InstancesById.Find(InstanceId);
	}
	else
	{
		Existing->LogicalAssetId = LogicalAssetId;
		Existing->PositionEnuM = PositionEnuM;
		Existing->RotationDeg = RotationDeg;
		Existing->QueryTags = QueryTags;
		Existing->EntityId = EntityId;
		Existing->MovementMode = TemplateDef->MovementMode;
		if (VisualState != nullptr)
		{
			Existing->VisualState = *VisualState;
			Existing->bHasVisualState = !VisualState->IsEmpty();
			Existing->bVisualStateExplicit = true;
		}
	}

	if (Existing == nullptr)
	{
		return false;
	}

	if (!SpawnActorForInstance(*Existing, OutError))
	{
		if (bCreatedInstance)
		{
			InstancesById.Remove(InstanceId);
		}
		return false;
	}

	return true;
}

bool UAeroAssetPlacementSubsystem::RemoveProxy(const FString& InstanceId, FString& OutError)
{
	FAeroAssetInstanceState* Instance = InstancesById.Find(InstanceId);
	if (Instance == nullptr)
	{
		OutError = FString::Printf(TEXT("Unknown asset_id: %s"), *InstanceId);
		return false;
	}

	DestroyActorForInstance(*Instance);
	if (Instance->bDynamic)
	{
		InstancesById.Remove(InstanceId);
	}
	else
	{
		Instance->bEnabled = false;
	}

	return true;
}

bool UAeroAssetPlacementSubsystem::ParseTemplateDefinition(const TSharedPtr<FJsonObject>& Object, FAeroAssetTemplateDefinition& OutTemplate, FString& OutError) const
{
	if (!Object->TryGetStringField(TEXT("logical_asset_id"), OutTemplate.LogicalAssetId))
	{
		OutError = TEXT("asset_catalog entry missing logical_asset_id.");
		return false;
	}
	if (!Object->TryGetStringField(TEXT("semantic_type"), OutTemplate.SemanticType))
	{
		OutError = FString::Printf(TEXT("asset_catalog entry '%s' missing semantic_type."), *OutTemplate.LogicalAssetId);
		return false;
	}
	if (!Object->TryGetStringField(TEXT("spawn_backend"), OutTemplate.SpawnBackend))
	{
		OutError = FString::Printf(TEXT("asset_catalog entry '%s' missing spawn_backend."), *OutTemplate.LogicalAssetId);
		return false;
	}

	Object->TryGetStringField(TEXT("ue_asset_path"), OutTemplate.UEAssetPath);
	TryReadVectorField(Object, TEXT("default_scale_xyz"), OutTemplate.DefaultScale);
	Object->TryGetNumberField(TEXT("default_yaw_offset_deg"), OutTemplate.DefaultYawOffsetDeg);
	Object->TryGetNumberField(TEXT("default_z_offset_m"), OutTemplate.DefaultZOffsetM);
	Object->TryGetStringField(TEXT("ground_snap_policy"), OutTemplate.GroundSnapPolicy);
	Object->TryGetBoolField(TEXT("physics_enabled"), OutTemplate.bPhysicsEnabled);
	Object->TryGetStringField(TEXT("airsim_registry_name"), OutTemplate.AirSimRegistryName);
	Object->TryGetBoolField(TEXT("is_blueprint"), OutTemplate.bIsBlueprint);
	Object->TryGetStringField(TEXT("collision_profile"), OutTemplate.CollisionProfile);
	Object->TryGetStringField(TEXT("feedback_mode"), OutTemplate.FeedbackMode);
	Object->TryGetStringField(TEXT("world_layer_type"), OutTemplate.WorldLayerType);
	Object->TryGetStringField(TEXT("zone_kind"), OutTemplate.ZoneKind);
	Object->TryGetStringField(TEXT("label_class"), OutTemplate.LabelClass);
	Object->TryGetBoolField(TEXT("render_required"), OutTemplate.bRenderRequired);
	if (!Object->HasField(TEXT("render_required")) &&
		OutTemplate.SpawnBackend.Equals(TEXT("semantic_only"), ESearchCase::IgnoreCase))
	{
		OutTemplate.bRenderRequired = false;
	}
	Object->TryGetBoolField(TEXT("annotation_visible"), OutTemplate.bAnnotationVisible);
	Object->TryGetBoolField(TEXT("reservable"), OutTemplate.bReservable);
	Object->TryGetBoolField(TEXT("blocking"), OutTemplate.bBlocking);
	FString MovementModeString;
	if (Object->TryGetStringField(TEXT("movement_mode"), MovementModeString))
	{
		OutTemplate.MovementMode = AeroParseMovementMode(MovementModeString);
	}
	OutTemplate.QueryTags = ReadStringArray(Object, TEXT("query_tags"));
	if (Object->HasTypedField<EJson::Object>(TEXT("default_visual_state")))
	{
		OutTemplate.bHasDefaultVisualState = AeroVisualStateFromJson(Object->GetObjectField(TEXT("default_visual_state")), OutTemplate.DefaultVisualState);
	}
	return true;
}

bool UAeroAssetPlacementSubsystem::ParseScenarioObject(const TSharedPtr<FJsonObject>& Object, FAeroAssetInstanceState& OutInstance, FString& OutError) const
{
	if (!Object->TryGetStringField(TEXT("instance_id"), OutInstance.InstanceId) &&
		!Object->TryGetStringField(TEXT("entity_id"), OutInstance.InstanceId))
	{
		OutError = TEXT("scenario_objects entry missing instance_id/entity_id.");
		return false;
	}
	if (!Object->TryGetStringField(TEXT("logical_asset_id"), OutInstance.LogicalAssetId))
	{
		OutError = FString::Printf(TEXT("scenario_objects entry '%s' missing logical_asset_id."), *OutInstance.InstanceId);
		return false;
	}
	if (!Object->TryGetStringField(TEXT("placement_mode"), OutInstance.PlacementMode))
	{
		OutError = FString::Printf(TEXT("scenario_objects entry '%s' missing placement_mode."), *OutInstance.InstanceId);
		return false;
	}
	if (!Object->HasTypedField<EJson::Object>(TEXT("placement")))
	{
		OutError = FString::Printf(TEXT("scenario_objects entry '%s' missing placement object."), *OutInstance.InstanceId);
		return false;
	}

	OutInstance.bEnabled = Object->HasField(TEXT("enabled")) ? Object->GetBoolField(TEXT("enabled")) : true;
	double ActivationTickValue = 0.0;
	if (Object->TryGetNumberField(TEXT("activation_tick"), ActivationTickValue))
	{
		OutInstance.ActivationTick = FMath::Max(0, static_cast<int32>(FMath::RoundToInt(ActivationTickValue)));
	}
	Object->TryGetStringField(TEXT("spawn_policy"), OutInstance.SpawnPolicy);
	if (OutInstance.ActivationTick > 0 || OutInstance.SpawnPolicy.Equals(TEXT("event_script_only"), ESearchCase::IgnoreCase))
	{
		OutInstance.bEnabled = false;
	}
	OutInstance.QueryTags = ReadStringArray(Object, TEXT("query_tags"));
	Object->TryGetStringField(TEXT("entity_id"), OutInstance.EntityId);
	if (OutInstance.EntityId.IsEmpty())
	{
		OutInstance.EntityId = OutInstance.InstanceId;
	}
	Object->TryGetStringField(TEXT("world_layer_type"), OutInstance.WorldLayerType);
	Object->TryGetStringField(TEXT("zone_kind"), OutInstance.ZoneKind);
	OutInstance.Placement = Object->GetObjectField(TEXT("placement"));
	OutInstance.bDynamic = false;
	if (Object->HasTypedField<EJson::Object>(TEXT("initial_state")))
	{
		OutInstance.InitialState = Object->GetObjectField(TEXT("initial_state"));
		if (!TryReadVisualStateField(OutInstance.InitialState, TEXT("visual_state"), OutInstance.VisualState))
		{
			OutInstance.bHasVisualState = AeroVisualStateFromJson(OutInstance.InitialState, OutInstance.VisualState);
		}
		else
		{
			OutInstance.bHasVisualState = !OutInstance.VisualState.IsEmpty();
		}
		OutInstance.bVisualStateExplicit = OutInstance.bHasVisualState;
	}
	return true;
}

bool UAeroAssetPlacementSubsystem::TryResolvePlacementPosition(const FAeroAssetInstanceState& Instance, FVector& OutPositionEnuM, FRotator& OutRotationDeg) const
{
	if (!Instance.Placement.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("TryResolvePlacementPosition failed for '%s': placement object is invalid."), *Instance.InstanceId);
		return false;
	}

	if (Instance.PlacementMode.Equals(TEXT("world_pose"), ESearchCase::IgnoreCase))
	{
		if (!TryReadVectorField(Instance.Placement, TEXT("resolved_position_enu_m"), OutPositionEnuM) &&
			!TryReadVectorField(Instance.Placement, TEXT("position_enu_m"), OutPositionEnuM) &&
			!TryReadVectorField(Instance.Placement, TEXT("center_enu_m"), OutPositionEnuM))
		{
			UE_LOG(LogTemp, Warning, TEXT("TryResolvePlacementPosition failed for '%s': world_pose missing position_enu_m/center_enu_m."), *Instance.InstanceId);
			return false;
		}
		TryReadRotationField(Instance.Placement, TEXT("rotation_deg"), OutRotationDeg);
		Instance.Placement->TryGetNumberField(TEXT("yaw_deg"), OutRotationDeg.Yaw);
		return true;
	}

	if (Instance.PlacementMode.Equals(TEXT("lane_anchor"), ESearchCase::IgnoreCase) ||
		Instance.PlacementMode.Equals(TEXT("sidewalk_anchor"), ESearchCase::IgnoreCase) ||
		Instance.PlacementMode.Equals(TEXT("crosswalk_anchor"), ESearchCase::IgnoreCase) ||
		Instance.PlacementMode.Equals(TEXT("facade_anchor"), ESearchCase::IgnoreCase) ||
		Instance.PlacementMode.Equals(TEXT("pad_anchor"), ESearchCase::IgnoreCase))
	{
		if (!TryReadVectorField(Instance.Placement, TEXT("resolved_position_enu_m"), OutPositionEnuM) &&
			!TryReadVectorField(Instance.Placement, TEXT("position_enu_m"), OutPositionEnuM) &&
			!TryReadVectorField(Instance.Placement, TEXT("center_enu_m"), OutPositionEnuM))
		{
			UE_LOG(LogTemp, Warning, TEXT("TryResolvePlacementPosition failed for '%s': semantic placement '%s' missing resolved_position_enu_m."), *Instance.InstanceId, *Instance.PlacementMode);
			return false;
		}
		TryReadRotationField(Instance.Placement, TEXT("rotation_deg"), OutRotationDeg);
		Instance.Placement->TryGetNumberField(TEXT("yaw_deg"), OutRotationDeg.Yaw);
		return true;
	}

	if (Instance.PlacementMode.Equals(TEXT("anchor_ref"), ESearchCase::IgnoreCase))
	{
		FString AnchorId;
		if (!Instance.Placement->TryGetStringField(TEXT("anchor_id"), AnchorId))
		{
			UE_LOG(LogTemp, Warning, TEXT("TryResolvePlacementPosition failed for '%s': anchor_ref missing anchor_id."), *Instance.InstanceId);
			return false;
		}

		const FAeroAssetInstanceState* Anchor = FindInstance(AnchorId);
		if (Anchor == nullptr)
		{
			UE_LOG(LogTemp, Warning, TEXT("TryResolvePlacementPosition failed for '%s': anchor_ref could not resolve anchor '%s'."), *Instance.InstanceId, *AnchorId);
			return false;
		}

		OutPositionEnuM = Anchor->PositionEnuM;
		OutRotationDeg = Anchor->RotationDeg;
		FVector OffsetEnuM = FVector::ZeroVector;
		TryReadVectorField(Instance.Placement, TEXT("offset_enu_m"), OffsetEnuM);
		OutPositionEnuM += OffsetEnuM;
		Instance.Placement->TryGetNumberField(TEXT("yaw_deg"), OutRotationDeg.Yaw);
		return true;
	}

	if (Instance.PlacementMode.Equals(TEXT("box_volume"), ESearchCase::IgnoreCase) || Instance.PlacementMode.Equals(TEXT("sphere_volume"), ESearchCase::IgnoreCase))
	{
		if (!TryReadVectorField(Instance.Placement, TEXT("resolved_position_enu_m"), OutPositionEnuM) &&
			!TryReadVectorField(Instance.Placement, TEXT("center_enu_m"), OutPositionEnuM) &&
			!TryReadVectorField(Instance.Placement, TEXT("position_enu_m"), OutPositionEnuM))
		{
			UE_LOG(LogTemp, Warning, TEXT("TryResolvePlacementPosition failed for '%s': volume placement missing center_enu_m/position_enu_m."), *Instance.InstanceId);
			return false;
		}
		TryReadRotationField(Instance.Placement, TEXT("rotation_deg"), OutRotationDeg);
		Instance.Placement->TryGetNumberField(TEXT("yaw_deg"), OutRotationDeg.Yaw);
		return true;
	}

	if (Instance.PlacementMode.Equals(TEXT("polygon_prism"), ESearchCase::IgnoreCase))
	{
		if (TryReadVectorField(Instance.Placement, TEXT("resolved_position_enu_m"), OutPositionEnuM))
		{
			OutRotationDeg = FRotator::ZeroRotator;
			return true;
		}
		TArray<FVector> PolygonEnuM;
		if (!TryReadPolygonArrayField(Instance.Placement, TEXT("polygon_enu_m"), PolygonEnuM) || PolygonEnuM.Num() < 3)
		{
			UE_LOG(LogTemp, Warning, TEXT("TryResolvePlacementPosition failed for '%s': polygon_prism missing or invalid polygon_enu_m."), *Instance.InstanceId);
			return false;
		}

		FVector Centroid = FVector::ZeroVector;
		for (const FVector& Vertex : PolygonEnuM)
		{
			Centroid += Vertex;
		}
		Centroid /= static_cast<double>(PolygonEnuM.Num());

		double MinZM = 0.0;
		double MaxZM = 0.0;
		if (!Instance.Placement->TryGetNumberField(TEXT("min_z_m"), MinZM))
		{
			double BaseZM = Centroid.Z;
			Instance.Placement->TryGetNumberField(TEXT("base_z_m"), BaseZM);
			double HeightM = 0.0;
			Instance.Placement->TryGetNumberField(TEXT("height_m"), HeightM);
			MinZM = BaseZM;
			MaxZM = BaseZM + HeightM;
		}
		else
		{
			Instance.Placement->TryGetNumberField(TEXT("max_z_m"), MaxZM);
		}

		OutPositionEnuM = FVector(Centroid.X, Centroid.Y, (MinZM + MaxZM) * 0.5);
		OutRotationDeg = FRotator::ZeroRotator;
		return true;
	}

	return false;
}

bool UAeroAssetPlacementSubsystem::SpawnScenarioActors(FString& OutError)
{
	int32 FailCount = 0;
	FString LastFailError;
	for (TPair<FString, FAeroAssetInstanceState>& Pair : InstancesById)
	{
		FAeroAssetInstanceState& Instance = Pair.Value;
		if (Instance.bDynamic || !Instance.bEnabled)
		{
			continue;
		}

		FString InstanceError;
		if (!SpawnActorForInstance(Instance, InstanceError))
		{
			UE_LOG(LogTemp, Warning, TEXT("SpawnScenarioActors: skipping instance '%s': %s"), *Instance.InstanceId, *InstanceError);
			Instance.bEnabled = false;
			LastFailError = MoveTemp(InstanceError);
			++FailCount;
		}
	}
	if (FailCount > 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("SpawnScenarioActors: %d instance(s) failed to spawn and were disabled."), FailCount);
	}
	return true;
}

bool UAeroAssetPlacementSubsystem::SpawnActorForInstance(FAeroAssetInstanceState& Instance, FString& OutError)
{
	const FAeroAssetTemplateDefinition* TemplateDef = TemplatesById.Find(Instance.LogicalAssetId);
	if (TemplateDef == nullptr)
	{
		OutError = FString::Printf(TEXT("Unknown logical_asset_id '%s' for instance '%s'."), *Instance.LogicalAssetId, *Instance.InstanceId);
		return false;
	}

	FVector PositionEnuM = Instance.PositionEnuM;
	FRotator RotationDeg = Instance.RotationDeg;
	if (!Instance.bDynamic)
	{
		if (!TryResolvePlacementPosition(Instance, PositionEnuM, RotationDeg))
		{
			OutError = FString::Printf(TEXT("Failed to resolve placement for instance '%s'."), *Instance.InstanceId);
			return false;
		}

		Instance.PositionEnuM = PositionEnuM;
		Instance.RotationDeg = RotationDeg;
	}

	UWorld* World = GetWorld();
	if (World == nullptr)
	{
		OutError = TEXT("No valid UWorld for asset spawning.");
		return false;
	}

	const FVector WorldLocationCm = TemplateDef->SpawnBackend.Equals(TEXT("trigger_zone"), ESearchCase::IgnoreCase)
		? ConvertEnuMetersToWorldCm(PositionEnuM)
		: ApplyGroundSnapAndOffsets(*TemplateDef, ConvertEnuMetersToWorldCm(PositionEnuM), &Instance.LastGroundSource);
	const FRotator FinalRotation = RotationDeg + FRotator(0.0, TemplateDef->DefaultYawOffsetDeg, 0.0);
	const FAeroSemanticBindingData BindingData = BuildBindingData(*TemplateDef, Instance);
	Instance.MovementMode = TemplateDef->MovementMode;
	Instance.LastResolvedWorldLocationCm = WorldLocationCm;

	if (TemplateDef->SpawnBackend.Equals(TEXT("semantic_only"), ESearchCase::IgnoreCase))
	{
		DestroyActorForInstance(Instance);
		if (TemplateDef->bRenderRequired)
		{
			OutError = FString::Printf(
				TEXT("Asset '%s' logical_asset_id='%s' requires rendering but spawn_backend='semantic_only'."),
				*Instance.InstanceId,
				*Instance.LogicalAssetId);
			return false;
		}
		return true;
	}

	if (!TemplateDef->SpawnBackend.Equals(TEXT("trigger_zone"), ESearchCase::IgnoreCase) && TemplateDef->UEAssetPath.IsEmpty())
	{
		DestroyActorForInstance(Instance);
		if (TemplateDef->bRenderRequired)
		{
			OutError = FString::Printf(
				TEXT("Asset '%s' logical_asset_id='%s' requires rendering but ue_asset_path is empty."),
				*Instance.InstanceId,
				*Instance.LogicalAssetId);
			return false;
		}
		return true;
	}

	if (Instance.Actor.IsValid())
	{
		MoveActorForTemplate(Instance.Actor.Get(), *TemplateDef, WorldLocationCm, FinalRotation);
		AlignActorToGroundByBounds(Instance.Actor.Get(), *TemplateDef);
		Instance.Actor->SetActorScale3D(TemplateDef->DefaultScale);
		Instance.PositionEnuM = ConvertWorldCmToEnuMeters(Instance.Actor->GetActorLocation());
		Instance.LastResolvedWorldLocationCm = Instance.Actor->GetActorLocation();
		Instance.RotationDeg = RotationDeg;
		FAeroSemanticRuntimeHelpers::ApplySemanticBinding(Instance.Actor.Get(), BindingData);
		ApplyInstanceVisualState(Instance.Actor.Get(), *TemplateDef, Instance);
		if (TemplateDef->SpawnBackend.Equals(TEXT("trigger_zone"), ESearchCase::IgnoreCase))
		{
			FAeroTriggerShapeConfig ShapeConfig;
			if (!TryBuildTriggerShapeConfig(Instance, PositionEnuM, ShapeConfig, OutError))
			{
				return false;
			}

			AAeroTriggerZoneBase* TriggerActor = Cast<AAeroTriggerZoneBase>(Instance.Actor.Get());
			if (!IsValid(TriggerActor))
			{
				OutError = FString::Printf(TEXT("Instance '%s' expected AAeroTriggerZoneBase actor."), *Instance.InstanceId);
				return false;
			}
			FAeroSemanticRuntimeHelpers::ConfigureTriggerActor(TriggerActor, BindingData, ShapeConfig);
		}
		return true;
	}

	FActorSpawnParameters SpawnParameters;
	SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;

	AActor* SpawnedActor = nullptr;
	if (TemplateDef->SpawnBackend.Equals(TEXT("trigger_zone"), ESearchCase::IgnoreCase))
	{
		if (TemplateDef->UEAssetPath.IsEmpty())
		{
			SpawnedActor = World->SpawnActor<AAeroTriggerZoneBase>(AAeroTriggerZoneBase::StaticClass(), WorldLocationCm, FinalRotation, SpawnParameters);
		}
		else
		{
			UObject* LoadedAsset = FSoftObjectPath(TemplateDef->UEAssetPath).TryLoad();
			if (LoadedAsset == nullptr)
			{
				OutError = FString::Printf(TEXT("Failed to load asset path '%s' for '%s'."), *TemplateDef->UEAssetPath, *Instance.InstanceId);
				return false;
			}

			if (UBlueprint* Blueprint = Cast<UBlueprint>(LoadedAsset))
			{
				SpawnedActor = World->SpawnActor<AActor>(Blueprint->GeneratedClass, WorldLocationCm, FinalRotation, SpawnParameters);
			}
			else if (UClass* ActorClass = Cast<UClass>(LoadedAsset))
			{
				SpawnedActor = World->SpawnActor<AActor>(ActorClass, WorldLocationCm, FinalRotation, SpawnParameters);
			}
		}
	}
	else if (!TemplateDef->UEAssetPath.IsEmpty())
	{
		UObject* LoadedAsset = FSoftObjectPath(TemplateDef->UEAssetPath).TryLoad();
		if (LoadedAsset == nullptr)
		{
			OutError = FString::Printf(TEXT("Failed to load asset path '%s' for '%s'."), *TemplateDef->UEAssetPath, *Instance.InstanceId);
			return false;
		}

		if (UBlueprint* Blueprint = Cast<UBlueprint>(LoadedAsset))
		{
			SpawnedActor = World->SpawnActor<AActor>(Blueprint->GeneratedClass, WorldLocationCm, FinalRotation, SpawnParameters);
		}
		else if (UClass* ActorClass = Cast<UClass>(LoadedAsset))
		{
			SpawnedActor = World->SpawnActor<AActor>(ActorClass, WorldLocationCm, FinalRotation, SpawnParameters);
		}
		else if (UStaticMesh* StaticMesh = Cast<UStaticMesh>(LoadedAsset))
		{
			AStaticMeshActor* MeshActor = World->SpawnActor<AStaticMeshActor>(WorldLocationCm, FinalRotation, SpawnParameters);
			if (MeshActor != nullptr)
			{
				MeshActor->SetMobility(EComponentMobility::Movable);
				UStaticMeshComponent* MeshComponent = MeshActor->GetStaticMeshComponent();
				if (MeshComponent != nullptr)
				{
					MeshComponent->SetMobility(EComponentMobility::Movable);
					MeshComponent->SetStaticMesh(StaticMesh);
					if (!TemplateDef->CollisionProfile.IsEmpty())
					{
						MeshComponent->SetCollisionProfileName(*TemplateDef->CollisionProfile);
					}
				}
				SpawnedActor = MeshActor;
			}
		}
	}

	if (SpawnedActor == nullptr)
	{
		OutError = FString::Printf(
			TEXT("Failed to spawn actor for instance '%s' logical_asset_id='%s' asset_path='%s' world='%s'."),
			*Instance.InstanceId,
			*Instance.LogicalAssetId,
			*TemplateDef->UEAssetPath,
			*WorldLocationCm.ToString());
		return false;
	}

	SpawnedActor->SetActorScale3D(TemplateDef->DefaultScale);
	if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(SpawnedActor->GetRootComponent()))
	{
		Primitive->SetSimulatePhysics(TemplateDef->bPhysicsEnabled);
		if (!TemplateDef->CollisionProfile.IsEmpty())
		{
			Primitive->SetCollisionProfileName(*TemplateDef->CollisionProfile);
		}
	}

	AlignActorToGroundByBounds(SpawnedActor, *TemplateDef);

	FAeroSemanticRuntimeHelpers::ApplySemanticBinding(SpawnedActor, BindingData);
	if (TemplateDef->FeedbackMode.Equals(TEXT("hit"), ESearchCase::IgnoreCase) || TemplateDef->FeedbackMode.Equals(TEXT("both"), ESearchCase::IgnoreCase))
	{
		FAeroSemanticRuntimeHelpers::EnsureCollisionRelay(SpawnedActor);
	}

	if (TemplateDef->SpawnBackend.Equals(TEXT("trigger_zone"), ESearchCase::IgnoreCase))
	{
		AAeroTriggerZoneBase* TriggerActor = Cast<AAeroTriggerZoneBase>(SpawnedActor);
		if (!IsValid(TriggerActor))
		{
			OutError = FString::Printf(TEXT("Trigger asset '%s' does not derive from AAeroTriggerZoneBase."), *Instance.InstanceId);
			SpawnedActor->Destroy();
			return false;
		}

		FAeroTriggerShapeConfig ShapeConfig;
		if (!TryBuildTriggerShapeConfig(Instance, PositionEnuM, ShapeConfig, OutError))
		{
			SpawnedActor->Destroy();
			return false;
		}

		FAeroSemanticRuntimeHelpers::ConfigureTriggerActor(TriggerActor, BindingData, ShapeConfig);
	}

	ApplyInstanceVisualState(SpawnedActor, *TemplateDef, Instance);
	Instance.Actor = SpawnedActor;
	Instance.PositionEnuM = ConvertWorldCmToEnuMeters(SpawnedActor->GetActorLocation());
	Instance.LastResolvedWorldLocationCm = SpawnedActor->GetActorLocation();
	Instance.RotationDeg = RotationDeg;
	return true;
}

void UAeroAssetPlacementSubsystem::DestroyActorForInstance(FAeroAssetInstanceState& Instance)
{
	if (Instance.Actor.IsValid())
	{
		Instance.Actor->Destroy();
		Instance.Actor.Reset();
	}
}

FAeroSemanticBindingData UAeroAssetPlacementSubsystem::BuildBindingData(const FAeroAssetTemplateDefinition& TemplateDef, const FAeroAssetInstanceState& Instance) const
{
	FAeroSemanticBindingData BindingData;
	BindingData.EntityId = Instance.EntityId.TrimStartAndEnd().IsEmpty() ? Instance.InstanceId : Instance.EntityId.TrimStartAndEnd();
	BindingData.InstanceId = Instance.InstanceId;
	BindingData.LogicalAssetId = Instance.LogicalAssetId;
	BindingData.Tags = TemplateDef.QueryTags;
	for (const FString& InstanceTag : Instance.QueryTags)
	{
		if (!BindingData.Tags.Contains(InstanceTag))
		{
			BindingData.Tags.Add(InstanceTag);
		}
	}
	BindingData.WorldLayerType = !Instance.WorldLayerType.IsEmpty() ? Instance.WorldLayerType : TemplateDef.WorldLayerType;
	BindingData.ZoneKind = !Instance.ZoneKind.IsEmpty() ? Instance.ZoneKind : TemplateDef.ZoneKind;
	BindingData.LabelClass = TemplateDef.LabelClass;
	BindingData.bRenderRequired = TemplateDef.bRenderRequired;
	BindingData.bAnnotationVisible = TemplateDef.bAnnotationVisible;
	BindingData.FeedbackMode = AeroParseFeedbackMode(TemplateDef.FeedbackMode);
	return BindingData;
}

bool UAeroAssetPlacementSubsystem::TryBuildTriggerShapeConfig(const FAeroAssetInstanceState& Instance, const FVector& OriginEnuM, FAeroTriggerShapeConfig& OutShapeConfig, FString& OutError) const
{
	OutShapeConfig = FAeroTriggerShapeConfig();
	OutShapeConfig.ShapeKind = AeroParseTriggerShapeKind(Instance.PlacementMode);
	if (!Instance.Placement.IsValid())
	{
		OutError = FString::Printf(TEXT("Trigger instance '%s' has invalid placement."), *Instance.InstanceId);
		return false;
	}

	if (OutShapeConfig.ShapeKind == EAeroTriggerShapeKind::Box)
	{
		FVector ExtentM = FVector::ZeroVector;
		if (TryReadVectorField(Instance.Placement, TEXT("extent_m"), ExtentM))
		{
			OutShapeConfig.BoxExtentCm = ExtentM * 100.0;
			return true;
		}

		FVector SizeM = FVector::ZeroVector;
		if (TryReadVectorField(Instance.Placement, TEXT("size_m"), SizeM))
		{
			OutShapeConfig.BoxExtentCm = SizeM * 50.0;
			return true;
		}

		OutError = FString::Printf(TEXT("Trigger instance '%s' missing extent_m/size_m."), *Instance.InstanceId);
		return false;
	}

	if (OutShapeConfig.ShapeKind == EAeroTriggerShapeKind::Sphere)
	{
		double RadiusM = 0.0;
		if (!Instance.Placement->TryGetNumberField(TEXT("radius_m"), RadiusM))
		{
			OutError = FString::Printf(TEXT("Trigger instance '%s' missing radius_m."), *Instance.InstanceId);
			return false;
		}

		OutShapeConfig.SphereRadiusCm = RadiusM * 100.0;
		return true;
	}

	if (OutShapeConfig.ShapeKind == EAeroTriggerShapeKind::PolygonPrism)
	{
		TArray<FVector> PolygonEnuM;
		if (!TryReadPolygonArrayField(Instance.Placement, TEXT("polygon_enu_m"), PolygonEnuM) || PolygonEnuM.Num() < 3)
		{
			OutError = FString::Printf(TEXT("Trigger instance '%s' missing polygon_enu_m vertices."), *Instance.InstanceId);
			return false;
		}

		OutShapeConfig.PolygonVerticesCm.Reset();
		for (const FVector& Vertex : PolygonEnuM)
		{
			const FVector LocalVertexCm = (Vertex - OriginEnuM) * 100.0;
			OutShapeConfig.PolygonVerticesCm.Add(FVector2D(LocalVertexCm.X, LocalVertexCm.Y));
		}

		double MinZM = 0.0;
		double MaxZM = 0.0;
		if (!Instance.Placement->TryGetNumberField(TEXT("min_z_m"), MinZM))
		{
			double BaseZM = OriginEnuM.Z;
			Instance.Placement->TryGetNumberField(TEXT("base_z_m"), BaseZM);
			double HeightM = 0.0;
			Instance.Placement->TryGetNumberField(TEXT("height_m"), HeightM);
			MinZM = BaseZM;
			MaxZM = BaseZM + HeightM;
		}
		else
		{
			Instance.Placement->TryGetNumberField(TEXT("max_z_m"), MaxZM);
		}

		OutShapeConfig.PolygonMinZCm = (MinZM - OriginEnuM.Z) * 100.0;
		OutShapeConfig.PolygonMaxZCm = (MaxZM - OriginEnuM.Z) * 100.0;
		return true;
	}

	OutError = FString::Printf(TEXT("Unsupported trigger placement mode '%s' for '%s'."), *Instance.PlacementMode, *Instance.InstanceId);
	return false;
}

FVector UAeroAssetPlacementSubsystem::ConvertEnuMetersToWorldCm(const FVector& PositionEnuM) const
{
	return CurrentWorldOriginCm + PositionEnuM * 100.0;
}

FVector UAeroAssetPlacementSubsystem::ConvertWorldCmToEnuMeters(const FVector& WorldLocationCm) const
{
	return (WorldLocationCm - CurrentWorldOriginCm) / 100.0;
}

FVector UAeroAssetPlacementSubsystem::ApplyGroundSnapAndOffsets(const FAeroAssetTemplateDefinition& TemplateDef, FVector WorldLocationCm) const
{
	return ApplyGroundSnapAndOffsets(TemplateDef, WorldLocationCm, nullptr);
}

FVector UAeroAssetPlacementSubsystem::ApplyGroundSnapAndOffsets(const FAeroAssetTemplateDefinition& TemplateDef, FVector WorldLocationCm, FString* OutGroundSource) const
{
	const FVector RequestedWorldCm = WorldLocationCm;
	if (OutGroundSource != nullptr)
	{
		OutGroundSource->Reset();
	}
	WorldLocationCm.Z += TemplateDef.DefaultZOffsetM * 100.0;
	if (!TemplateDef.GroundSnapPolicy.Equals(TEXT("project_down"), ESearchCase::IgnoreCase))
	{
		return WorldLocationCm;
	}

	const UWorld* World = GetWorld();
	if (World == nullptr)
	{
		return WorldLocationCm;
	}

	AeroGroundPlacement::FResolvedGroundPlacement Placement;
	if (AeroGroundPlacement::ResolveGroundPlacement(const_cast<UWorld*>(World), RequestedWorldCm, Placement))
	{
		WorldLocationCm.Z = Placement.GroundWorldCm.Z + TemplateDef.DefaultZOffsetM * 100.0;
		if (OutGroundSource != nullptr)
		{
			*OutGroundSource = Placement.Source;
		}
	}
	return WorldLocationCm;
}

bool UAeroAssetPlacementSubsystem::MoveActorForTemplate(AActor* Actor, const FAeroAssetTemplateDefinition& TemplateDef, const FVector& WorldLocationCm, const FRotator& FinalRotation) const
{
	if (!IsValid(Actor))
	{
		return false;
	}

	if (TemplateDef.MovementMode == EAeroMovementMode::SweepFollow)
	{
		if (Cast<UPrimitiveComponent>(Actor->GetRootComponent()) != nullptr)
		{
			const FVector StartWorldLocationCm = Actor->GetActorLocation();
			FHitResult SweepHit;
			const bool bMoved = Actor->SetActorLocationAndRotation(WorldLocationCm, FinalRotation, true, &SweepHit, ETeleportType::None);
			if (SweepHit.bBlockingHit)
			{
				MaybeEmitSweepCollision(Actor, SweepHit, StartWorldLocationCm);
			}

			return bMoved;
		}
	}

	return Actor->SetActorLocationAndRotation(WorldLocationCm, FinalRotation, false, nullptr, ETeleportType::TeleportPhysics);
}

void UAeroAssetPlacementSubsystem::AlignActorToGroundByBounds(AActor* Actor, const FAeroAssetTemplateDefinition& TemplateDef) const
{
	if (!IsValid(Actor) || GetWorld() == nullptr)
	{
		return;
	}

	if (!TemplateDef.GroundSnapPolicy.Equals(TEXT("project_down"), ESearchCase::IgnoreCase) ||
		TemplateDef.SpawnBackend.Equals(TEXT("trigger_zone"), ESearchCase::IgnoreCase) ||
		TemplateDef.SpawnBackend.Equals(TEXT("semantic_only"), ESearchCase::IgnoreCase) ||
		TemplateDef.SemanticType.Contains(TEXT("uav"), ESearchCase::IgnoreCase) ||
		TemplateDef.LogicalAssetId.StartsWith(TEXT("uav.")))
	{
		return;
	}

	FVector BoundsOrigin = FVector::ZeroVector;
	FVector BoundsExtent = FVector::ZeroVector;
	Actor->GetActorBounds(true, BoundsOrigin, BoundsExtent);
	FVector GroundPoint = BoundsOrigin;
	if (!AeroGroundPlacement::TryProjectWorldPointToGround(GetWorld(), BoundsOrigin, GroundPoint, nullptr, Actor))
	{
		return;
	}

	const float BottomZ = BoundsOrigin.Z - BoundsExtent.Z;
	const float DeltaZ = GroundPoint.Z - BottomZ;
	if (FMath::Abs(DeltaZ) <= KINDA_SMALL_NUMBER)
	{
		return;
	}

	Actor->AddActorWorldOffset(FVector(0.0f, 0.0f, DeltaZ), false, nullptr, ETeleportType::TeleportPhysics);
}

void UAeroAssetPlacementSubsystem::MaybeEmitSweepCollision(AActor* Actor, const FHitResult& SweepHit, const FVector& StartWorldLocationCm) const
{
	if (!IsValid(Actor) || !SweepHit.bBlockingHit)
	{
		return;
	}

	AActor* OtherActor = SweepHit.GetActor();
	if (!IsValid(OtherActor) || OtherActor == Actor)
	{
		return;
	}

	FAeroSemanticBindingData SelfBinding;
	if (!FAeroSemanticRuntimeHelpers::ResolveSemanticBinding(Actor, SelfBinding) ||
		(SelfBinding.FeedbackMode != EAeroFeedbackMode::Hit && SelfBinding.FeedbackMode != EAeroFeedbackMode::Both))
	{
		return;
	}

	UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>() : nullptr;
	if (FeedbackSubsystem == nullptr)
	{
		return;
	}

	FAeroFeedbackEvent Event;
	Event.Type = TEXT("collision");
	FAeroSemanticRuntimeHelpers::CopySemanticBindingToFeedback(Actor, true, Event);
	FAeroSemanticRuntimeHelpers::CopySemanticBindingToFeedback(OtherActor, false, Event);
	Event.SourceActorId = Actor->GetName();
	Event.OtherActorId = OtherActor->GetName();
	Event.Collision.ContactPointEnuM = FeedbackSubsystem->WorldCmToEnuM(
		SweepHit.ImpactPoint.IsNearlyZero() ? SweepHit.Location : SweepHit.ImpactPoint);
	Event.Collision.ContactNormalEnu = FVector(SweepHit.ImpactNormal);
	const double WorldDeltaSeconds = GetWorld() != nullptr ? static_cast<double>(GetWorld()->GetDeltaSeconds()) : 0.0;
	const double SelfSpeedMps = WorldDeltaSeconds > KINDA_SMALL_NUMBER
		? FVector::Distance(StartWorldLocationCm, Actor->GetActorLocation()) / 100.0 / WorldDeltaSeconds
		: Actor->GetVelocity().Size() / 100.0;
	const double OtherSpeedMps = OtherActor->GetVelocity().Size() / 100.0;
	Event.Collision.RelativeSpeedMps = FMath::Max(SelfSpeedMps, OtherSpeedMps);
	Event.Collision.Impulse = 0.0;
	Event.Collision.bBlocking = true;
	FeedbackSubsystem->EnqueueFeedback(MoveTemp(Event));
}

void UAeroAssetPlacementSubsystem::ApplyInstanceVisualState(AActor* Actor, const FAeroAssetTemplateDefinition& TemplateDef, FAeroAssetInstanceState& Instance) const
{
	if (!IsValid(Actor))
	{
		return;
	}

	const FAeroVisualState* VisualState = nullptr;
	if (Instance.bVisualStateExplicit)
	{
		VisualState = &Instance.VisualState;
	}
	else if (TemplateDef.bHasDefaultVisualState)
	{
		VisualState = &TemplateDef.DefaultVisualState;
	}

	if (VisualState != nullptr)
	{
		FAeroSemanticRuntimeHelpers::ApplyVisualState(Actor, *VisualState);
	}
}

bool UAeroAssetPlacementSubsystem::TryReadVisualStateField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FAeroVisualState& OutState) const
{
	if (!Object.IsValid())
	{
		return false;
	}

	if (FieldName.TrimStartAndEnd().IsEmpty())
	{
		AeroVisualStateFromJson(Object, OutState);
		return true;
	}

	if (!Object->HasTypedField<EJson::Object>(FieldName))
	{
		return false;
	}

	AeroVisualStateFromJson(Object->GetObjectField(FieldName), OutState);
	return true;
}

bool UAeroAssetPlacementSubsystem::TryReadVectorField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutVector, double DefaultZ) const
{
	if (!Object.IsValid())
	{
		return false;
	}

	if (Object->HasTypedField<EJson::Array>(FieldName))
	{
		const TArray<TSharedPtr<FJsonValue>>& Values = Object->GetArrayField(FieldName);
		if (Values.Num() >= 2)
		{
			OutVector.X = Values[0]->AsNumber();
			OutVector.Y = Values[1]->AsNumber();
			OutVector.Z = Values.Num() > 2 ? Values[2]->AsNumber() : DefaultZ;
			return true;
		}
	}

	if (Object->HasTypedField<EJson::Object>(FieldName))
	{
		const TSharedPtr<FJsonObject> VectorObject = Object->GetObjectField(FieldName);
		if (VectorObject.IsValid())
		{
			const double X = VectorObject->HasField(TEXT("x")) ? VectorObject->GetNumberField(TEXT("x")) : VectorObject->GetNumberField(TEXT("east_m"));
			const double Y = VectorObject->HasField(TEXT("y")) ? VectorObject->GetNumberField(TEXT("y")) : VectorObject->GetNumberField(TEXT("north_m"));
			const double Z = VectorObject->HasField(TEXT("z")) ? VectorObject->GetNumberField(TEXT("z")) : (VectorObject->HasField(TEXT("up_m")) ? VectorObject->GetNumberField(TEXT("up_m")) : DefaultZ);
			OutVector = FVector(X, Y, Z);
			return true;
		}
	}

	return false;
}

bool UAeroAssetPlacementSubsystem::TryReadRotationField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FRotator& OutRotation) const
{
	if (!Object.IsValid() || !Object->HasTypedField<EJson::Object>(FieldName))
	{
		return false;
	}

	const TSharedPtr<FJsonObject> RotationObject = Object->GetObjectField(FieldName);
	if (!RotationObject.IsValid())
	{
		return false;
	}

	RotationObject->TryGetNumberField(TEXT("roll_deg"), OutRotation.Roll);
	RotationObject->TryGetNumberField(TEXT("pitch_deg"), OutRotation.Pitch);
	RotationObject->TryGetNumberField(TEXT("yaw_deg"), OutRotation.Yaw);
	return true;
}

bool UAeroAssetPlacementSubsystem::ReadPoseField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutPositionEnuM, FRotator& OutRotation) const
{
	if (!Object.IsValid() || !Object->HasTypedField<EJson::Object>(FieldName))
	{
		return false;
	}

	const TSharedPtr<FJsonObject> PoseObject = Object->GetObjectField(FieldName);
	if (!PoseObject.IsValid())
	{
		return false;
	}

	const bool bHasPosition = TryReadVectorField(PoseObject, TEXT("position_m"), OutPositionEnuM) || TryReadVectorField(PoseObject, TEXT("position_enu_m"), OutPositionEnuM);
	TryReadRotationField(PoseObject, TEXT("rotation_deg"), OutRotation);
	PoseObject->TryGetNumberField(TEXT("yaw_deg"), OutRotation.Yaw);
	return bHasPosition;
}

bool UAeroAssetPlacementSubsystem::ReadWorldPoseField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutPositionWorldCm, FRotator& OutRotation) const
{
	if (!Object.IsValid() || !Object->HasTypedField<EJson::Object>(FieldName))
	{
		return false;
	}

	const TSharedPtr<FJsonObject> PoseObject = Object->GetObjectField(FieldName);
	if (!PoseObject.IsValid())
	{
		return false;
	}

	const bool bHasPosition = TryReadVectorField(PoseObject, TEXT("position_world_cm"), OutPositionWorldCm) ||
		TryReadVectorField(PoseObject, TEXT("position_cm"), OutPositionWorldCm);
	TryReadRotationField(PoseObject, TEXT("rotation_deg"), OutRotation);
	PoseObject->TryGetNumberField(TEXT("yaw_deg"), OutRotation.Yaw);
	return bHasPosition;
}

bool UAeroAssetPlacementSubsystem::TryResolvePayloadPose(
	const TSharedPtr<FJsonObject>& Object,
	FVector& OutPositionEnuM,
	FVector* OutPositionWorldCm,
	FRotator& OutRotation) const
{
	FVector ResolvedWorldCm = FVector::ZeroVector;
	if (ReadPoseField(Object, TEXT("pose_enu_m"), OutPositionEnuM, OutRotation))
	{
		if (OutPositionWorldCm != nullptr)
		{
			*OutPositionWorldCm = ConvertEnuMetersToWorldCm(OutPositionEnuM);
		}
		return true;
	}

	if (ReadWorldPoseField(Object, TEXT("pose_world_cm"), ResolvedWorldCm, OutRotation))
	{
		OutPositionEnuM = ConvertWorldCmToEnuMeters(ResolvedWorldCm);
		if (OutPositionWorldCm != nullptr)
		{
			*OutPositionWorldCm = ResolvedWorldCm;
		}
		return true;
	}

	if (TryReadVectorField(Object, TEXT("position_enu_m"), OutPositionEnuM))
	{
		Object->TryGetNumberField(TEXT("yaw_deg"), OutRotation.Yaw);
		if (OutPositionWorldCm != nullptr)
		{
			*OutPositionWorldCm = ConvertEnuMetersToWorldCm(OutPositionEnuM);
		}
		return true;
	}

	if (TryReadVectorField(Object, TEXT("position_world_cm"), ResolvedWorldCm))
	{
		Object->TryGetNumberField(TEXT("yaw_deg"), OutRotation.Yaw);
		OutPositionEnuM = ConvertWorldCmToEnuMeters(ResolvedWorldCm);
		if (OutPositionWorldCm != nullptr)
		{
			*OutPositionWorldCm = ResolvedWorldCm;
		}
		return true;
	}

	return false;
}

bool UAeroAssetPlacementSubsystem::TryReadPolygonArrayField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, TArray<FVector>& OutPolygonEnuM) const
{
	OutPolygonEnuM.Reset();
	if (!Object.IsValid() || !Object->HasTypedField<EJson::Array>(FieldName))
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>& Values = Object->GetArrayField(FieldName);
	for (const TSharedPtr<FJsonValue>& Value : Values)
	{
		if (!Value.IsValid())
		{
			continue;
		}

		if (Value->Type == EJson::Array)
		{
			const TArray<TSharedPtr<FJsonValue>>& PointValues = Value->AsArray();
			if (PointValues.Num() >= 2)
			{
				OutPolygonEnuM.Add(FVector(
					PointValues[0]->AsNumber(),
					PointValues[1]->AsNumber(),
					PointValues.Num() > 2 ? PointValues[2]->AsNumber() : 0.0));
			}
			continue;
		}

		if (Value->Type == EJson::Object)
		{
			const TSharedPtr<FJsonObject> PointObject = Value->AsObject();
			if (PointObject.IsValid())
			{
				FVector Point = FVector::ZeroVector;
				if (!TryReadVectorField(PointObject, TEXT("point_enu_m"), Point) &&
					!TryReadVectorField(PointObject, TEXT("position_enu_m"), Point) &&
					!TryReadVectorField(PointObject, TEXT("value"), Point))
				{
					continue;
				}
				OutPolygonEnuM.Add(Point);
			}
		}
	}

	return OutPolygonEnuM.Num() >= 3;
}
