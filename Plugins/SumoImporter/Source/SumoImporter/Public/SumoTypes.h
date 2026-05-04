#pragma once

#include "CoreMinimal.h"
#include "SumoTypes.generated.h"

UENUM(BlueprintType)
enum class ESumoAxisMapping : uint8
{
	XY_To_XY UMETA(DisplayName = "SUMO XY -> UE XY"),
	XY_To_XNegY UMETA(DisplayName = "SUMO XY -> UE X(-Y)"),
	XY_To_YX UMETA(DisplayName = "SUMO XY -> UE YX"),
	XY_To_YNegX UMETA(DisplayName = "SUMO XY -> UE Y(-X)")
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoParseOptions
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SUMO")
	bool bImportInternalEdges = false;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoLocationData
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString NetOffset;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString ConvBoundary;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString OrigBoundary;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString ProjParameter;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoLaneData
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString EdgeId;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString LaneId;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	int32 LaneIndex = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	float SpeedMps = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	float LengthM = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	TArray<FVector> ShapePointsM;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoEdgeData
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString EdgeId;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString FromJunction;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString ToJunction;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString Function = TEXT("normal");

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString EdgeType;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	TArray<FSumoLaneData> Lanes;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoJunctionData
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString JunctionId;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FVector CenterM = FVector::ZeroVector;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	TArray<FVector> PolygonPointsM;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoConnectionData
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString FromEdge;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString ToEdge;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	int32 FromLane = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	int32 ToLane = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FString ViaLane;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoNetData
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	FSumoLocationData Location;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	TArray<FSumoEdgeData> Edges;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	TArray<FSumoJunctionData> Junctions;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO")
	TArray<FSumoConnectionData> Connections;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoTransformConfig
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SUMO|Transform")
	float ScaleCmPerMeter = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SUMO|Transform")
	ESumoAxisMapping AxisMapping = ESumoAxisMapping::XY_To_YX;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SUMO|Transform")
	float YawOffsetDeg = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SUMO|Transform")
	FVector TranslationCm = FVector::ZeroVector;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoBuildOptions
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SUMO|Build")
	bool bReplaceExistingActor = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SUMO|Build")
	bool bBuildJunctionDebug = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SUMO|Build")
	FName NetworkActorName = TEXT("SUMO_RoadNetwork");
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoImportStats
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	int32 TotalEdges = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	int32 ImportedEdges = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	int32 ImportedLanes = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	int32 SkippedEdges = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	int32 SkippedLanes = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	int32 JunctionCount = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	int32 ConnectionCount = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	int32 WarningCount = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Stats")
	TArray<FString> Warnings;

	void AddWarning(const FString& Warning)
	{
		++WarningCount;
		if (Warnings.Num() < 64)
		{
			Warnings.Add(Warning);
		}
	}
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoLaneHandle
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	FString LaneId;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	FString EdgeId;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	int32 LaneIndex = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	float LengthM = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	float SpeedMps = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	FString EdgeType;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	float RoadWeight = 1.0f;
};

USTRUCT(BlueprintType)
struct SUMOIMPORTER_API FSumoNearestLaneSample
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	FString LaneId;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	FString EdgeId;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	int32 LaneIndex = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	float Distance2DCm = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	float DistanceAlongLaneM = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SUMO|Query")
	FTransform WorldTransform = FTransform::Identity;
};
