#include "AeroSemanticTypes.h"

#include "Dom/JsonObject.h"

namespace
{
void SetStringArrayField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const TArray<FString>& Values)
{
	TArray<TSharedPtr<FJsonValue>> JsonValues;
	for (const FString& Value : Values)
	{
		JsonValues.Add(MakeShared<FJsonValueString>(Value));
	}
	Object->SetArrayField(FieldName, JsonValues);
}

void SetVectorArrayField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, const FVector& Value)
{
	TArray<TSharedPtr<FJsonValue>> JsonValues;
	JsonValues.Add(MakeShared<FJsonValueNumber>(Value.X));
	JsonValues.Add(MakeShared<FJsonValueNumber>(Value.Y));
	JsonValues.Add(MakeShared<FJsonValueNumber>(Value.Z));
	Object->SetArrayField(FieldName, JsonValues);
}

FName ReadStateNameField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName)
{
	FString RawValue;
	if (!Object.IsValid() || !Object->TryGetStringField(FieldName, RawValue))
	{
		return NAME_None;
	}

	RawValue = RawValue.TrimStartAndEnd();
	return RawValue.IsEmpty() ? NAME_None : FName(*RawValue);
}
}

EAeroFeedbackMode AeroParseFeedbackMode(const FString& Value)
{
	if (Value.Equals(TEXT("hit"), ESearchCase::IgnoreCase))
	{
		return EAeroFeedbackMode::Hit;
	}
	if (Value.Equals(TEXT("overlap"), ESearchCase::IgnoreCase))
	{
		return EAeroFeedbackMode::Overlap;
	}
	if (Value.Equals(TEXT("both"), ESearchCase::IgnoreCase))
	{
		return EAeroFeedbackMode::Both;
	}
	return EAeroFeedbackMode::None;
}

FString AeroFeedbackModeToString(EAeroFeedbackMode Value)
{
	switch (Value)
	{
	case EAeroFeedbackMode::Hit:
		return TEXT("hit");
	case EAeroFeedbackMode::Overlap:
		return TEXT("overlap");
	case EAeroFeedbackMode::Both:
		return TEXT("both");
	default:
		return TEXT("none");
	}
}

EAeroTriggerShapeKind AeroParseTriggerShapeKind(const FString& Value)
{
	if (Value.Equals(TEXT("box"), ESearchCase::IgnoreCase) || Value.Equals(TEXT("box_volume"), ESearchCase::IgnoreCase))
	{
		return EAeroTriggerShapeKind::Box;
	}
	if (Value.Equals(TEXT("sphere"), ESearchCase::IgnoreCase) || Value.Equals(TEXT("sphere_volume"), ESearchCase::IgnoreCase))
	{
		return EAeroTriggerShapeKind::Sphere;
	}
	if (Value.Equals(TEXT("polygon_prism"), ESearchCase::IgnoreCase))
	{
		return EAeroTriggerShapeKind::PolygonPrism;
	}
	return EAeroTriggerShapeKind::None;
}

FString AeroTriggerShapeKindToString(EAeroTriggerShapeKind Value)
{
	switch (Value)
	{
	case EAeroTriggerShapeKind::Box:
		return TEXT("box");
	case EAeroTriggerShapeKind::Sphere:
		return TEXT("sphere");
	case EAeroTriggerShapeKind::PolygonPrism:
		return TEXT("polygon_prism");
	default:
		return TEXT("none");
	}
}

EAeroMovementMode AeroParseMovementMode(const FString& Value)
{
	if (Value.Equals(TEXT("sweep_follow"), ESearchCase::IgnoreCase))
	{
		return EAeroMovementMode::SweepFollow;
	}

	return EAeroMovementMode::Teleport;
}

FString AeroMovementModeToString(EAeroMovementMode Value)
{
	switch (Value)
	{
	case EAeroMovementMode::SweepFollow:
		return TEXT("sweep_follow");
	default:
		return TEXT("teleport");
	}
}

bool AeroVisualStateFromJson(const TSharedPtr<FJsonObject>& Object, FAeroVisualState& OutState)
{
	OutState = FAeroVisualState();
	if (!Object.IsValid())
	{
		return false;
	}

	Object->TryGetStringField(TEXT("mode"), OutState.Mode);
	OutState.VariantId = ReadStateNameField(Object, TEXT("variant_id"));
	OutState.MontageTag = ReadStateNameField(Object, TEXT("montage_tag"));
	if (Object->HasField(TEXT("lights_on")))
	{
		OutState.bHasLightsOn = Object->TryGetBoolField(TEXT("lights_on"), OutState.bLightsOn);
	}
	Object->TryGetStringField(TEXT("material_variant"), OutState.MaterialVariant);

	for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : Object->Values)
	{
		if (Pair.Key.Equals(TEXT("mode"), ESearchCase::IgnoreCase) ||
			Pair.Key.Equals(TEXT("variant_id"), ESearchCase::IgnoreCase) ||
			Pair.Key.Equals(TEXT("montage_tag"), ESearchCase::IgnoreCase) ||
			Pair.Key.Equals(TEXT("lights_on"), ESearchCase::IgnoreCase) ||
			Pair.Key.Equals(TEXT("material_variant"), ESearchCase::IgnoreCase))
		{
			continue;
		}

		UE_LOG(LogTemp, Verbose, TEXT("AeroVisualStateFromJson ignored unsupported key '%s'."), *Pair.Key);
	}

	return !OutState.IsEmpty();
}

TSharedPtr<FJsonObject> AeroVisualStateToJson(const FAeroVisualState& State)
{
	TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
	if (!State.Mode.TrimStartAndEnd().IsEmpty())
	{
		Root->SetStringField(TEXT("mode"), State.Mode);
	}
	if (!State.VariantId.IsNone())
	{
		Root->SetStringField(TEXT("variant_id"), State.VariantId.ToString());
	}
	if (!State.MontageTag.IsNone())
	{
		Root->SetStringField(TEXT("montage_tag"), State.MontageTag.ToString());
	}
	if (State.bHasLightsOn)
	{
		Root->SetBoolField(TEXT("lights_on"), State.bLightsOn);
	}
	if (!State.MaterialVariant.TrimStartAndEnd().IsEmpty())
	{
		Root->SetStringField(TEXT("material_variant"), State.MaterialVariant);
	}

	return Root;
}

TSharedPtr<FJsonObject> AeroFeedbackEventToJson(const FAeroFeedbackEvent& Event)
{
	TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
	Root->SetStringField(TEXT("type"), Event.Type);
	Root->SetStringField(TEXT("event_id"), Event.EventId);
	Root->SetNumberField(TEXT("tick"), Event.Tick);
	Root->SetNumberField(TEXT("frame_id"), Event.FrameId);
	Root->SetStringField(TEXT("episode_id"), Event.EpisodeId);
	if (Event.SampleSeq != INDEX_NONE)
	{
		Root->SetNumberField(TEXT("sample_seq"), Event.SampleSeq);
	}
	Root->SetNumberField(TEXT("sim_time_s"), Event.SimTimeS);
	Root->SetStringField(TEXT("source_entity_id"), Event.SourceEntityId);
	Root->SetStringField(TEXT("other_entity_id"), Event.OtherEntityId);
	Root->SetStringField(TEXT("source_actor_id"), Event.SourceActorId);
	Root->SetStringField(TEXT("other_actor_id"), Event.OtherActorId);
	Root->SetStringField(TEXT("source_logical_asset_id"), Event.SourceLogicalAssetId);
	Root->SetStringField(TEXT("other_logical_asset_id"), Event.OtherLogicalAssetId);
	SetStringArrayField(Root, TEXT("source_tags"), Event.SourceTags);
	SetStringArrayField(Root, TEXT("other_tags"), Event.OtherTags);

	if (Event.Type.Equals(TEXT("collision"), ESearchCase::IgnoreCase))
	{
		SetVectorArrayField(Root, TEXT("contact_point_enu_m"), Event.Collision.ContactPointEnuM);
		SetVectorArrayField(Root, TEXT("contact_normal_enu"), Event.Collision.ContactNormalEnu);
		Root->SetNumberField(TEXT("relative_speed_mps"), Event.Collision.RelativeSpeedMps);
		Root->SetNumberField(TEXT("impulse"), Event.Collision.Impulse);
		Root->SetBoolField(TEXT("blocking"), Event.Collision.bBlocking);
	}
	else
	{
		Root->SetStringField(TEXT("world_layer_type"), Event.Overlap.WorldLayerType);
		Root->SetStringField(TEXT("zone_kind"), Event.Overlap.ZoneKind);
	}

	return Root;
}
