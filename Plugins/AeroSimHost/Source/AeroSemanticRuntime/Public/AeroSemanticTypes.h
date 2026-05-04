#pragma once

#include "CoreMinimal.h"
#include "AeroSemanticTypes.generated.h"

class FJsonObject;

UENUM(BlueprintType)
enum class EAeroFeedbackMode : uint8
{
	None,
	Hit,
	Overlap,
	Both
};

UENUM(BlueprintType)
enum class EAeroTriggerShapeKind : uint8
{
	None,
	Box,
	Sphere,
	PolygonPrism
};

UENUM(BlueprintType)
enum class EAeroMovementMode : uint8
{
	Teleport,
	SweepFollow
};

USTRUCT(BlueprintType)
struct AEROSEMANTICRUNTIME_API FAeroFrameContext
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	int64 Tick = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	int64 FrameId = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString EpisodeId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	int64 SampleSeq = INDEX_NONE;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	double SimTimeS = 0.0;
};

USTRUCT(BlueprintType)
struct AEROSEMANTICRUNTIME_API FAeroVisualState
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString Mode;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FName VariantId = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FName MontageTag = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	bool bHasLightsOn = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	bool bLightsOn = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString MaterialVariant;

	bool IsEmpty() const
	{
		return Mode.TrimStartAndEnd().IsEmpty() && VariantId.IsNone() && MontageTag.IsNone() && !bHasLightsOn && MaterialVariant.TrimStartAndEnd().IsEmpty();
	}
};

USTRUCT(BlueprintType)
struct AEROSEMANTICRUNTIME_API FAeroSemanticBindingData
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString EntityId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString InstanceId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString LogicalAssetId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	TArray<FString> Tags;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString WorldLayerType;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString ZoneKind;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString LabelClass;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	bool bRenderRequired = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	bool bAnnotationVisible = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	EAeroFeedbackMode FeedbackMode = EAeroFeedbackMode::None;
};

USTRUCT(BlueprintType)
struct AEROSEMANTICRUNTIME_API FAeroTriggerShapeConfig
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	EAeroTriggerShapeKind ShapeKind = EAeroTriggerShapeKind::None;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FVector BoxExtentCm = FVector(50.0f, 50.0f, 50.0f);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	float SphereRadiusCm = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	TArray<FVector2D> PolygonVerticesCm;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	float PolygonMinZCm = -50.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	float PolygonMaxZCm = 50.0f;
};

USTRUCT(BlueprintType)
struct AEROSEMANTICRUNTIME_API FAeroCollisionFeedback
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FVector ContactPointEnuM = FVector::ZeroVector;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FVector ContactNormalEnu = FVector::ZeroVector;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	double RelativeSpeedMps = 0.0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	double Impulse = 0.0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	bool bBlocking = true;
};

USTRUCT(BlueprintType)
struct AEROSEMANTICRUNTIME_API FAeroOverlapFeedback
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString WorldLayerType;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString ZoneKind;
};

USTRUCT(BlueprintType)
struct AEROSEMANTICRUNTIME_API FAeroFeedbackEvent
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString Type;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString EventId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	int64 Tick = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	int64 FrameId = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString EpisodeId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	int64 SampleSeq = INDEX_NONE;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	double SimTimeS = 0.0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString SourceEntityId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString OtherEntityId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString SourceActorId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString OtherActorId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString SourceLogicalAssetId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FString OtherLogicalAssetId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	TArray<FString> SourceTags;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	TArray<FString> OtherTags;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FAeroCollisionFeedback Collision;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero")
	FAeroOverlapFeedback Overlap;
};

AEROSEMANTICRUNTIME_API EAeroFeedbackMode AeroParseFeedbackMode(const FString& Value);
AEROSEMANTICRUNTIME_API FString AeroFeedbackModeToString(EAeroFeedbackMode Value);
AEROSEMANTICRUNTIME_API EAeroTriggerShapeKind AeroParseTriggerShapeKind(const FString& Value);
AEROSEMANTICRUNTIME_API FString AeroTriggerShapeKindToString(EAeroTriggerShapeKind Value);
AEROSEMANTICRUNTIME_API EAeroMovementMode AeroParseMovementMode(const FString& Value);
AEROSEMANTICRUNTIME_API FString AeroMovementModeToString(EAeroMovementMode Value);
AEROSEMANTICRUNTIME_API bool AeroVisualStateFromJson(const TSharedPtr<FJsonObject>& Object, FAeroVisualState& OutState);
AEROSEMANTICRUNTIME_API TSharedPtr<FJsonObject> AeroVisualStateToJson(const FAeroVisualState& State);
AEROSEMANTICRUNTIME_API TSharedPtr<FJsonObject> AeroFeedbackEventToJson(const FAeroFeedbackEvent& Event);
