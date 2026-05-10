#include "AeroSceneSyncSubsystem.h"

#include "AeroAssetPlacementSubsystem.h"
#include "AeroFeedbackSubsystem.h"
#include "AeroSemanticTypes.h"
#include "Dom/JsonObject.h"

namespace
{
bool ReadVisualStateField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FAeroVisualState& OutVisualState)
{
	if (!Object.IsValid() || !Object->HasTypedField<EJson::Object>(FieldName))
	{
		return false;
	}

	AeroVisualStateFromJson(Object->GetObjectField(FieldName), OutVisualState);
	return true;
}

void WriteVectorArrayField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FVector& Value)
{
	TArray<TSharedPtr<FJsonValue>> Values;
	Values.Add(MakeShared<FJsonValueNumber>(Value.X));
	Values.Add(MakeShared<FJsonValueNumber>(Value.Y));
	Values.Add(MakeShared<FJsonValueNumber>(Value.Z));
	Object->SetArrayField(FieldName, Values);
}

void WriteVectorObjectField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FVector& Value)
{
	TSharedPtr<FJsonObject> VectorObject = MakeShared<FJsonObject>();
	VectorObject->SetNumberField(TEXT("x"), Value.X);
	VectorObject->SetNumberField(TEXT("y"), Value.Y);
	VectorObject->SetNumberField(TEXT("z"), Value.Z);
	Object->SetObjectField(FieldName, VectorObject);
}

void AppendResolvedProxyState(const UAeroAssetPlacementSubsystem* AssetSubsystem, const FString& ProxyId, const TSharedPtr<FJsonObject>& Result)
{
	if (AssetSubsystem == nullptr || !Result.IsValid())
	{
		return;
	}

	const FAeroAssetInstanceState* ResolvedInstance = AssetSubsystem->FindInstance(ProxyId);
	if (ResolvedInstance == nullptr)
	{
		return;
	}

	Result->SetStringField(TEXT("logical_asset_id"), ResolvedInstance->LogicalAssetId);
	WriteVectorArrayField(Result, TEXT("position_enu_m"), ResolvedInstance->PositionEnuM);
	WriteVectorObjectField(Result, TEXT("position_world_cm"), ResolvedInstance->LastResolvedWorldLocationCm);
	if (ResolvedInstance->Actor.IsValid())
	{
		Result->SetStringField(TEXT("actor_name"), ResolvedInstance->Actor->GetName());
	}

	if (!ResolvedInstance->LastGroundSource.IsEmpty())
	{
		Result->SetStringField(TEXT("ground_source"), ResolvedInstance->LastGroundSource);
	}
}
}

bool UAeroSceneSyncSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

TSharedPtr<FJsonObject> UAeroSceneSyncSubsystem::ApplyFrame(const TSharedPtr<FJsonObject>& Payload, FString& OutError)
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("ApplyFrame payload is invalid.");
		return nullptr;
	}

	TArray<TSharedPtr<FJsonValue>> SpawnResults;
	TArray<TSharedPtr<FJsonValue>> UpdateResults;
	TArray<TSharedPtr<FJsonValue>> RemoveResults;

	if (UAeroFeedbackSubsystem* FeedbackSubsystem = GetWorld() != nullptr ? GetWorld()->GetSubsystem<UAeroFeedbackSubsystem>() : nullptr)
	{
		FAeroFrameContext FrameContext;
		double NumberValue = 0.0;
		if (Payload->TryGetNumberField(TEXT("tick"), NumberValue))
		{
			FrameContext.Tick = static_cast<int64>(NumberValue);
		}
		if (Payload->TryGetNumberField(TEXT("frame_id"), NumberValue))
		{
			FrameContext.FrameId = static_cast<int64>(NumberValue);
		}
		if (Payload->TryGetNumberField(TEXT("sample_seq"), NumberValue))
		{
			FrameContext.SampleSeq = static_cast<int64>(NumberValue);
		}
		Payload->TryGetNumberField(TEXT("sim_time_s"), FrameContext.SimTimeS);
		Payload->TryGetStringField(TEXT("episode_id"), FrameContext.EpisodeId);
		FeedbackSubsystem->SetFrameContext(FrameContext);
	}

	const TArray<TSharedPtr<FJsonValue>>* Spawns = nullptr;
	if (Payload->TryGetArrayField(TEXT("spawns"), Spawns) && Spawns != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *Spawns)
		{
			if (!ApplySpawnDelta(Value->AsObject(), SpawnResults, OutError))
			{
				return nullptr;
			}
		}
	}

	const TArray<TSharedPtr<FJsonValue>>* Updates = nullptr;
	if (Payload->TryGetArrayField(TEXT("updates"), Updates) && Updates != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *Updates)
		{
			if (!ApplyUpdateDelta(Value->AsObject(), UpdateResults, OutError))
			{
				return nullptr;
			}
		}
	}

	const TArray<TSharedPtr<FJsonValue>>* Removes = nullptr;
	if (Payload->TryGetArrayField(TEXT("removes"), Removes) && Removes != nullptr)
	{
		for (const TSharedPtr<FJsonValue>& Value : *Removes)
		{
			if (!ApplyRemoveDelta(Value->AsObject(), RemoveResults, OutError))
			{
				return nullptr;
			}
		}
	}

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetNumberField(TEXT("tick"), Payload->HasField(TEXT("tick")) ? Payload->GetNumberField(TEXT("tick")) : 0.0);
	Result->SetNumberField(TEXT("frame_id"), Payload->HasField(TEXT("frame_id")) ? Payload->GetNumberField(TEXT("frame_id")) : 0.0);
	Result->SetNumberField(TEXT("sim_time_s"), Payload->HasField(TEXT("sim_time_s")) ? Payload->GetNumberField(TEXT("sim_time_s")) : 0.0);
	if (Payload->HasField(TEXT("sample_seq")))
	{
		Result->SetNumberField(TEXT("sample_seq"), Payload->GetNumberField(TEXT("sample_seq")));
	}
	if (Payload->HasTypedField<EJson::String>(TEXT("episode_id")))
	{
		Result->SetStringField(TEXT("episode_id"), Payload->GetStringField(TEXT("episode_id")));
	}
	Result->SetArrayField(TEXT("spawns"), SpawnResults);
	Result->SetArrayField(TEXT("updates"), UpdateResults);
	Result->SetArrayField(TEXT("removes"), RemoveResults);
	return Result;
}

void UAeroSceneSyncSubsystem::ResetSyncState()
{
	EntityToProxyInstance.Reset();
}

bool UAeroSceneSyncSubsystem::ApplySpawnDelta(const TSharedPtr<FJsonObject>& DeltaObject, TArray<TSharedPtr<FJsonValue>>& OutSpawnResults, FString& OutError)
{
	if (!DeltaObject.IsValid())
	{
		OutError = TEXT("Spawn delta is invalid.");
		return false;
	}

	FString EntityId;
	FString ProxyTemplateId;
	if (!DeltaObject->TryGetStringField(TEXT("entity_id"), EntityId))
	{
		OutError = TEXT("Spawn delta missing entity_id.");
		return false;
	}
	if (!DeltaObject->TryGetStringField(TEXT("proxy_template_id"), ProxyTemplateId))
	{
		OutError = FString::Printf(TEXT("Spawn delta '%s' missing proxy_template_id."), *EntityId);
		return false;
	}

	FVector PositionEnuM = FVector::ZeroVector;
	FRotator RotationDeg = FRotator::ZeroRotator;
	if (!ReadPoseField(DeltaObject, TEXT("pose_enu_m"), PositionEnuM, RotationDeg))
	{
		OutError = FString::Printf(TEXT("Spawn delta '%s' missing pose_enu_m."), *EntityId);
		return false;
	}

	TArray<FString> QueryTags;
	ReadTagsField(DeltaObject, TEXT("tags"), QueryTags);
	FAeroVisualState VisualState;
	const bool bHasVisualState = ReadVisualStateField(DeltaObject, TEXT("visual_state"), VisualState);
	FString PlacementMode;
	const bool bHasPlacementMode = DeltaObject->TryGetStringField(TEXT("placement_mode"), PlacementMode);
	TSharedPtr<FJsonObject> Placement;
	const bool bHasPlacement = DeltaObject->HasTypedField<EJson::Object>(TEXT("placement"));
	if (bHasPlacement)
	{
		Placement = DeltaObject->GetObjectField(TEXT("placement"));
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("AeroAssetPlacement subsystem unavailable.");
		return false;
	}

	const FString ProxyId = FString::Printf(TEXT("entity_proxy_%s"), *EntityId);
	if (!AssetSubsystem->SpawnOrUpdateProxy(
			ProxyId,
			ProxyTemplateId,
			PositionEnuM,
			RotationDeg,
			QueryTags,
			EntityId,
			bHasVisualState ? &VisualState : nullptr,
			nullptr,
			false,
			OutError,
			bHasPlacementMode ? &PlacementMode : nullptr,
			bHasPlacement ? &Placement : nullptr))
	{
		return false;
	}

	EntityToProxyInstance.Add(EntityId, ProxyId);
	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("entity_id"), EntityId);
	Result->SetStringField(TEXT("proxy_id"), ProxyId);
	AppendResolvedProxyState(AssetSubsystem, ProxyId, Result);
	OutSpawnResults.Add(MakeShared<FJsonValueObject>(Result));
	return true;
}

bool UAeroSceneSyncSubsystem::ApplyUpdateDelta(const TSharedPtr<FJsonObject>& DeltaObject, TArray<TSharedPtr<FJsonValue>>& OutUpdateResults, FString& OutError)
{
	if (!DeltaObject.IsValid())
	{
		OutError = TEXT("Update delta is invalid.");
		return false;
	}

	FString EntityId;
	if (!DeltaObject->TryGetStringField(TEXT("entity_id"), EntityId))
	{
		OutError = TEXT("Update delta missing entity_id.");
		return false;
	}

	const FString* ProxyId = EntityToProxyInstance.Find(EntityId);
	if (ProxyId == nullptr)
	{
		return ApplySpawnDelta(DeltaObject, OutUpdateResults, OutError);
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("AeroAssetPlacement subsystem unavailable.");
		return false;
	}

	const FAeroAssetInstanceState* ExistingInstance = AssetSubsystem->FindInstance(*ProxyId);
	if (ExistingInstance == nullptr)
	{
		EntityToProxyInstance.Remove(EntityId);
		return ApplySpawnDelta(DeltaObject, OutUpdateResults, OutError);
	}

	FVector PositionEnuM = ExistingInstance->PositionEnuM;
	FRotator RotationDeg = ExistingInstance->RotationDeg;
	ReadPoseField(DeltaObject, TEXT("pose_enu_m"), PositionEnuM, RotationDeg);
	FAeroVisualState VisualState;
	const bool bHasVisualState = ReadVisualStateField(DeltaObject, TEXT("visual_state"), VisualState);
	FString PlacementMode = ExistingInstance->PlacementMode;
	DeltaObject->TryGetStringField(TEXT("placement_mode"), PlacementMode);
	TSharedPtr<FJsonObject> Placement = ExistingInstance->Placement;
	if (DeltaObject->HasTypedField<EJson::Object>(TEXT("placement")))
	{
		Placement = DeltaObject->GetObjectField(TEXT("placement"));
	}

	TArray<FString> QueryTags = ExistingInstance->QueryTags;
	ReadTagsField(DeltaObject, TEXT("tags"), QueryTags);
	if (!AssetSubsystem->SpawnOrUpdateProxy(
			*ProxyId,
			ExistingInstance->LogicalAssetId,
			PositionEnuM,
			RotationDeg,
			QueryTags,
			EntityId,
			bHasVisualState ? &VisualState : nullptr,
			ExistingInstance->bHasInstanceScale ? &ExistingInstance->InstanceScale : nullptr,
			ExistingInstance->bCustomStencilOnly,
			OutError,
			&PlacementMode,
			&Placement))
	{
		return false;
	}

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("entity_id"), EntityId);
	Result->SetStringField(TEXT("proxy_id"), *ProxyId);
	AppendResolvedProxyState(AssetSubsystem, *ProxyId, Result);
	OutUpdateResults.Add(MakeShared<FJsonValueObject>(Result));
	return true;
}

bool UAeroSceneSyncSubsystem::ApplyRemoveDelta(const TSharedPtr<FJsonObject>& DeltaObject, TArray<TSharedPtr<FJsonValue>>& OutRemoveResults, FString& OutError)
{
	if (!DeltaObject.IsValid())
	{
		OutError = TEXT("Remove delta is invalid.");
		return false;
	}

	FString EntityId;
	if (!DeltaObject->TryGetStringField(TEXT("entity_id"), EntityId))
	{
		OutError = TEXT("Remove delta missing entity_id.");
		return false;
	}

	const FString* ProxyId = EntityToProxyInstance.Find(EntityId);
	if (ProxyId == nullptr)
	{
		TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("entity_id"), EntityId);
		Result->SetBoolField(TEXT("removed"), false);
		OutRemoveResults.Add(MakeShared<FJsonValueObject>(Result));
		return true;
	}

	UAeroAssetPlacementSubsystem* AssetSubsystem = GetWorld()->GetSubsystem<UAeroAssetPlacementSubsystem>();
	if (AssetSubsystem == nullptr)
	{
		OutError = TEXT("AeroAssetPlacement subsystem unavailable.");
		return false;
	}

	if (!AssetSubsystem->RemoveProxy(*ProxyId, OutError))
	{
		return false;
	}

	EntityToProxyInstance.Remove(EntityId);
	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("entity_id"), EntityId);
	Result->SetBoolField(TEXT("removed"), true);
	OutRemoveResults.Add(MakeShared<FJsonValueObject>(Result));
	return true;
}

bool UAeroSceneSyncSubsystem::ReadPoseField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutPositionEnuM, FRotator& OutRotationDeg) const
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

	const TArray<TSharedPtr<FJsonValue>>* PositionValues = nullptr;
	if ((PoseObject->TryGetArrayField(TEXT("position_m"), PositionValues) || PoseObject->TryGetArrayField(TEXT("position_enu_m"), PositionValues)) &&
		PositionValues != nullptr && PositionValues->Num() >= 3)
	{
		OutPositionEnuM.X = (*PositionValues)[0]->AsNumber();
		OutPositionEnuM.Y = (*PositionValues)[1]->AsNumber();
		OutPositionEnuM.Z = (*PositionValues)[2]->AsNumber();
	}

	if (PoseObject->HasTypedField<EJson::Object>(TEXT("rotation_deg")))
	{
		const TSharedPtr<FJsonObject> RotationObject = PoseObject->GetObjectField(TEXT("rotation_deg"));
		RotationObject->TryGetNumberField(TEXT("roll_deg"), OutRotationDeg.Roll);
		RotationObject->TryGetNumberField(TEXT("pitch_deg"), OutRotationDeg.Pitch);
		RotationObject->TryGetNumberField(TEXT("yaw_deg"), OutRotationDeg.Yaw);
	}
	else
	{
		PoseObject->TryGetNumberField(TEXT("yaw_deg"), OutRotationDeg.Yaw);
	}
	return true;
}

void UAeroSceneSyncSubsystem::ReadTagsField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, TArray<FString>& OutTags) const
{
	if (!Object.IsValid())
	{
		return;
	}

	const TArray<TSharedPtr<FJsonValue>>* TagValues = nullptr;
	if (!Object->TryGetArrayField(FieldName, TagValues) || TagValues == nullptr)
	{
		return;
	}

	OutTags.Reset();
	for (const TSharedPtr<FJsonValue>& TagValue : *TagValues)
	{
		FString Tag;
		if (TagValue.IsValid() && TagValue->TryGetString(Tag))
		{
			OutTags.Add(Tag);
		}
	}
}
