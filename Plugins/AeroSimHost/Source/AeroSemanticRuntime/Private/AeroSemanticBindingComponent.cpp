#include "AeroSemanticBindingComponent.h"

UAeroSemanticBindingComponent::UAeroSemanticBindingComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UAeroSemanticBindingComponent::ConfigureFromData(const FAeroSemanticBindingData& InData)
{
	EntityId = InData.EntityId;
	InstanceId = InData.InstanceId;
	LogicalAssetId = InData.LogicalAssetId;
	Tags = InData.Tags;
	WorldLayerType = InData.WorldLayerType;
	ZoneKind = InData.ZoneKind;
	LabelClass = InData.LabelClass;
	bRenderRequired = InData.bRenderRequired;
	bAnnotationVisible = InData.bAnnotationVisible;
	FeedbackMode = InData.FeedbackMode;
}

FAeroSemanticBindingData UAeroSemanticBindingComponent::MakeBindingData() const
{
	FAeroSemanticBindingData Data;
	Data.EntityId = EntityId;
	Data.InstanceId = InstanceId;
	Data.LogicalAssetId = LogicalAssetId;
	Data.Tags = Tags;
	Data.WorldLayerType = WorldLayerType;
	Data.ZoneKind = ZoneKind;
	Data.LabelClass = LabelClass;
	Data.bRenderRequired = bRenderRequired;
	Data.bAnnotationVisible = bAnnotationVisible;
	Data.FeedbackMode = FeedbackMode;
	return Data;
}

FString UAeroSemanticBindingComponent::GetStableEntityId() const
{
	if (!EntityId.TrimStartAndEnd().IsEmpty())
	{
		return EntityId.TrimStartAndEnd();
	}
	if (!InstanceId.TrimStartAndEnd().IsEmpty())
	{
		return InstanceId.TrimStartAndEnd();
	}
	return FString();
}

bool UAeroSemanticBindingComponent::SupportsHitFeedback() const
{
	return FeedbackMode == EAeroFeedbackMode::Hit || FeedbackMode == EAeroFeedbackMode::Both;
}

bool UAeroSemanticBindingComponent::SupportsOverlapFeedback() const
{
	return FeedbackMode == EAeroFeedbackMode::Overlap || FeedbackMode == EAeroFeedbackMode::Both;
}
