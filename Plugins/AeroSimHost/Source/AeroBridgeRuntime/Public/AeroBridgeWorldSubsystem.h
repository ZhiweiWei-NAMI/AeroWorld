#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroBridgeWorldSubsystem.generated.h"

class FJsonObject;

UCLASS()
class AEROBRIDGERUNTIME_API UAeroBridgeWorldSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;

	UFUNCTION()
	FString HandleDescribeCapabilities(const FString& RequestJson);
	UFUNCTION()
	FString HandleLoadContext(const FString& RequestJson);
	UFUNCTION()
	FString HandleReloadConfig(const FString& RequestJson);
	UFUNCTION()
	FString HandleApplyFrame(const FString& RequestJson);
	UFUNCTION()
	FString HandlePollFeedback(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedSpawn(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedReset(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedSetTarget(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedObserve(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedPlayAnimation(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedCommitCross(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedStop(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedSetVariant(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedRelease(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedSpawnCrowd(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedClearCrowd(const FString& RequestJson);
	UFUNCTION()
	FString HandlePedRespawnCrowd(const FString& RequestJson);
	UFUNCTION()
	FString HandleSpawnAsset(const FString& RequestJson);
	UFUNCTION()
	FString HandleMoveAsset(const FString& RequestJson);
	UFUNCTION()
	FString HandleRemoveAsset(const FString& RequestJson);
	UFUNCTION()
	FString HandleCaptureWorldCamera(const FString& RequestJson);
	UFUNCTION()
	FString HandleReserveOccupancy(const FString& RequestJson);
	UFUNCTION()
	FString HandleReleaseOccupancy(const FString& RequestJson);
	UFUNCTION()
	FString HandleQueryNearest(const FString& RequestJson);
	UFUNCTION()
	FString HandleQueryPedPath(const FString& RequestJson);
	UFUNCTION()
	FString HandleProjectGround(const FString& RequestJson);
	UFUNCTION()
	FString HandleQueryPedAnchor(const FString& RequestJson);
	UFUNCTION()
	FString HandleApplyWeather(const FString& RequestJson);
	UFUNCTION()
	FString HandleCreateRuntimeMultirotor(const FString& RequestJson);
	UFUNCTION()
	FString HandleMoveRuntimeMultirotor(const FString& RequestJson);
	UFUNCTION()
	FString HandleGetRuntimeMultirotorStatus(const FString& RequestJson);
	UFUNCTION()
	FString HandleRemoveRuntimeVehicle(const FString& RequestJson);
	UFUNCTION()
	FString HandleGetRuntimeVehiclePose(const FString& RequestJson);

	const FString& GetCurrentMapId() const
	{
		return CurrentMapId;
	}

	const TSharedPtr<FJsonObject>& GetCurrentMapContext() const
	{
		return CurrentMapContext;
	}

private:
	TSharedPtr<FJsonObject> ParseRequestEnvelope(const FString& RequestJson, FString& OutRequestId, FString& OutMapId, FString& OutError) const;
	FString MakeSuccessResponse(const FString& Op, const FString& RequestId, const FString& MapId, const TSharedPtr<FJsonObject>& Payload) const;
	FString MakeErrorResponse(const FString& Op, const FString& RequestId, const FString& MapId, const FString& ErrorMessage) const;
	FString ResolveRelativePath(const FString& MaybeRelativePath) const;
	FString ResolveMapPath(const FString& MapId, const FString& FileName) const;
	bool LoadContextByMapId(const FString& MapId, FString& OutError);
	bool ApplyLoadedMapContext(const FString& MapId, const TSharedPtr<FJsonObject>& MapContext, FString& OutError);

private:
	FString CurrentMapId;
	TSharedPtr<FJsonObject> CurrentMapContext;
};
