#include "SumoSceneBuilder.h"

#include "Engine/World.h"
#include "EngineUtils.h"
#include "SumoCoordinateTransformer.h"
#include "SumoImporterLog.h"
#include "SumoRoadNetworkActor.h"

namespace
{
float ComputeRoadWeight(const FString& EdgeType, float SpeedMps)
{
	const FString LowerType = EdgeType.ToLower();
	if (LowerType.Contains(TEXT("motorway")) || LowerType.Contains(TEXT("expressway")) || LowerType.Contains(TEXT("freeway")))
	{
		return 1.8f;
	}
	if (LowerType.Contains(TEXT("trunk")))
	{
		return 1.6f;
	}
	if (LowerType.Contains(TEXT("primary")))
	{
		return 1.4f;
	}
	if (LowerType.Contains(TEXT("secondary")))
	{
		return 1.2f;
	}
	if (LowerType.Contains(TEXT("tertiary")))
	{
		return 1.0f;
	}
	if (LowerType.Contains(TEXT("residential")) || LowerType.Contains(TEXT("living_street")))
	{
		return 0.8f;
	}
	if (LowerType.Contains(TEXT("service")))
	{
		return 0.6f;
	}

	if (SpeedMps >= 25.0f)
	{
		return 1.6f;
	}
	if (SpeedMps >= 18.0f)
	{
		return 1.3f;
	}
	if (SpeedMps >= 12.0f)
	{
		return 1.1f;
	}
	if (SpeedMps >= 6.0f)
	{
		return 0.9f;
	}
	return 0.7f;
}
}

ASumoRoadNetworkActor* FSumoSceneBuilder::BuildToWorld(
	UWorld* World,
	const FSumoNetData& NetData,
	const FSumoTransformConfig& TransformConfig,
	const FSumoBuildOptions& BuildOptions,
	FSumoImportStats& InOutStats,
	FString& OutError)
{
	OutError.Reset();

	if (World == nullptr)
	{
		OutError = TEXT("Invalid world.");
		return nullptr;
	}

	if (BuildOptions.bReplaceExistingActor)
	{
		const FName NetworkTag = BuildOptions.NetworkActorName;
		for (TActorIterator<ASumoRoadNetworkActor> It(World); It; ++It)
		{
			if (It->ActorHasTag(NetworkTag))
			{
				It->Destroy();
			}
		}
	}

	FActorSpawnParameters SpawnParameters;
	SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
	ASumoRoadNetworkActor* NetworkActor = World->SpawnActor<ASumoRoadNetworkActor>(FVector::ZeroVector, FRotator::ZeroRotator, SpawnParameters);
	if (!IsValid(NetworkActor))
	{
		OutError = TEXT("Failed to spawn ASumoRoadNetworkActor.");
		return nullptr;
	}

#if WITH_EDITOR
	NetworkActor->SetActorLabel(BuildOptions.NetworkActorName.ToString());
#endif
	NetworkActor->Tags.AddUnique(BuildOptions.NetworkActorName);
	NetworkActor->ResetNetwork();

	for (const FSumoEdgeData& EdgeData : NetData.Edges)
	{
		for (const FSumoLaneData& LaneData : EdgeData.Lanes)
		{
			TArray<FVector> LanePointsWorld;
			FSumoCoordinateTransformer::TransformPointsMeters(LaneData.ShapePointsM, TransformConfig, LanePointsWorld);
			const float RoadWeight = ComputeRoadWeight(EdgeData.EdgeType, LaneData.SpeedMps);

			if (!NetworkActor->AddLane(LaneData, EdgeData.EdgeType, RoadWeight, LanePointsWorld))
			{
				InOutStats.SkippedLanes++;
			}
		}
	}

	if (BuildOptions.bBuildJunctionDebug)
	{
		for (const FSumoJunctionData& JunctionData : NetData.Junctions)
		{
			TArray<FVector> PolygonWorld;
			FSumoCoordinateTransformer::TransformPointsMeters(JunctionData.PolygonPointsM, TransformConfig, PolygonWorld);
			NetworkActor->AddJunctionDebugPolygon(JunctionData, PolygonWorld);
		}
	}

	NetworkActor->SetConnections(NetData.Connections);

	UE_LOG(
		LogSumoImporter,
		Log,
		TEXT("Built SUMO actor '%s': lanes=%d, junctions=%d."),
		*NetworkActor->GetName(),
		NetworkActor->GetLaneCount(),
		NetData.Junctions.Num());

	return NetworkActor;
}
