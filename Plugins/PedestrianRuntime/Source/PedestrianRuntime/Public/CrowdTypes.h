#pragma once

#include "CoreMinimal.h"
#include "Engine/EngineTypes.h"
#include "UObject/SoftObjectPtr.h"
#include "CrowdTypes.generated.h"

class UCrowdAppearancePool;
class UCrowdRoleProfile;
class UStaticMesh;

UENUM(BlueprintType)
enum class ECrowdGender : uint8
{
	Unknown = 0,
	Male,
	Female
};

UENUM(BlueprintType)
enum class ECrowdAgeGroup : uint8
{
	Unknown = 0,
	Child,
	Adult,
	Elder
};

UENUM(BlueprintType)
enum class ECrowdYawPolicy : uint8
{
	Random = 0,
	Fixed
};

USTRUCT(BlueprintType)
struct FCrowdAccessorySpec
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Accessory")
	FName AccessoryTag = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Accessory")
	TSoftObjectPtr<UStaticMesh> Mesh;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Accessory")
	FName SocketName = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Accessory", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float Probability = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Accessory")
	FVector RelativeLocation = FVector::ZeroVector;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Accessory")
	FRotator RelativeRotation = FRotator::ZeroRotator;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Accessory")
	FVector RelativeScale = FVector::OneVector;
};

USTRUCT(BlueprintType)
struct FCrowdAppearanceEntry
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	FName AppearanceId = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	FName VariantId = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	ECrowdGender Gender = ECrowdGender::Unknown;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	ECrowdAgeGroup AgeGroup = ECrowdAgeGroup::Unknown;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance", meta = (ClampMin = "0.0"))
	float Weight = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	FVector2D ScaleRange = FVector2D(1.0f, 1.0f);

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	FString MaterialVariant;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	TArray<FName> AccessoryTags;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	TArray<FCrowdAccessorySpec> OptionalAccessories;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Appearance")
	TArray<FName> SpawnTags;
};

USTRUCT(BlueprintType)
struct FCrowdTagWeightMultiplier
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role")
	FName Tag = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Role", meta = (ClampMin = "0.0"))
	float Multiplier = 1.0f;
};

USTRUCT(BlueprintType)
struct FCrowdSpawnRequest
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	FName GroupId = FName(TEXT("crowd.default"));

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn", meta = (ClampMin = "0"))
	int32 Count = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	int32 Seed = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	TObjectPtr<UCrowdAppearancePool> AppearancePool = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	TObjectPtr<UCrowdRoleProfile> RoleProfile = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	FVector SpawnOrigin = FVector::ZeroVector;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	FVector SpawnBoxExtent = FVector(500.0f, 500.0f, 0.0f);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	ECrowdYawPolicy YawPolicy = ECrowdYawPolicy::Random;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	float FixedYawDeg = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	bool bUseProvidedGroundPoint = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Crowd|Spawn")
	ESpawnActorCollisionHandlingMethod CollisionHandling = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
};

USTRUCT(BlueprintType)
struct FCrowdSpawnResult
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Spawn")
	FName GroupId = NAME_None;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Spawn")
	TArray<FString> SpawnedIds;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Spawn")
	int32 SkippedCount = 0;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Crowd|Spawn")
	int32 Seed = 0;
};
