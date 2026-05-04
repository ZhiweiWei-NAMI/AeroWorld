#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AeroSemanticTypes.h"
#include "AeroSemanticBindingComponent.generated.h"

UCLASS(ClassGroup = (Aero), BlueprintType, Blueprintable, meta = (BlueprintSpawnableComponent))
class AEROSEMANTICRUNTIME_API UAeroSemanticBindingComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAeroSemanticBindingComponent();

	void ConfigureFromData(const FAeroSemanticBindingData& InData);
	FAeroSemanticBindingData MakeBindingData() const;
	FString GetStableEntityId() const;
	bool SupportsHitFeedback() const;
	bool SupportsOverlapFeedback() const;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	FString EntityId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	FString InstanceId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	FString LogicalAssetId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	TArray<FString> Tags;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	FString WorldLayerType;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	FString ZoneKind;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	FString LabelClass;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	bool bRenderRequired = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	bool bAnnotationVisible = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Aero|Semantic")
	EAeroFeedbackMode FeedbackMode = EAeroFeedbackMode::None;
};
