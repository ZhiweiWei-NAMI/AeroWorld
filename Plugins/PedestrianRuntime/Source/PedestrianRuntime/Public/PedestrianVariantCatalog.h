#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "UObject/SoftObjectPtr.h"
#include "PedestrianVariantCatalog.generated.h"

class UAnimMontage;
class USkeletalMesh;

USTRUCT(BlueprintType)
struct FPedVariantSpec
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	FName VariantId = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	TSoftObjectPtr<USkeletalMesh> SkeletalMesh;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	float CapsuleRadius = -1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	float CapsuleHalfHeight = -1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	float DefaultWalkSpeed = -1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	FVector MeshRelativeLocationOffset = FVector::ZeroVector;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	FRotator MeshRelativeRotation = FRotator::ZeroRotator;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	FVector MeshRelativeScale = FVector::OneVector;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	float GroundContactOffsetCm = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant|Animation")
	TSoftObjectPtr<UAnimMontage> ObserveMontageOverride;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant|Animation")
	TSoftObjectPtr<UAnimMontage> StartCrossMontageOverride;
};

UCLASS(BlueprintType)
class PEDESTRIANRUNTIME_API UPedestrianVariantCatalog : public UDataAsset
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrian|Variant")
	TArray<FPedVariantSpec> Variants;

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Variant")
	bool FindVariantById(FName VariantId, FPedVariantSpec& OutSpec) const;

	const FPedVariantSpec* FindVariantById(FName VariantId) const;
};
