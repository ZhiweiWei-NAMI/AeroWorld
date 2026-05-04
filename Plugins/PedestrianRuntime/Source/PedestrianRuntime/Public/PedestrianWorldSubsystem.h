#pragma once

#include "CoreMinimal.h"
#include "CrowdTypes.h"
#include "Subsystems/WorldSubsystem.h"
#include "PedestrianWorldSubsystem.generated.h"

class APedestrianCharacter;
class UCrowdAppearancePool;
class UCrowdRoleProfile;

struct FCrowdRuntimeSelection
{
	FCrowdAppearanceEntry Appearance;
	float EffectiveWeight = 0.0f;
};

struct FCrowdGroupState
{
	FCrowdSpawnRequest Request;
	TArray<FString> SpawnedIds;
	int32 LastSeed = 0;
};

UCLASS()
class PEDESTRIANRUNTIME_API UPedestrianWorldSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual void Deinitialize() override;

	void RegisterPedestrian(APedestrianCharacter* Ped);
	void UnregisterPedestrian(APedestrianCharacter* Ped);
	APedestrianCharacter* FindPedestrian(const FString& PedId) const;

	bool ExecReset(const FString& PedId, const FVector& Loc, float YawDeg, bool bUseProvidedGroundPoint = false);
	bool ExecObserve(const FString& PedId);
	bool ExecCommitCross(const FString& PedId, const FVector& Target, float SpeedCmPerSec);
	bool ExecStop(const FString& PedId);
	bool ExecSetTarget(const FString& PedId, const FVector& Target, float SpeedCmPerSec);
	bool ExecSetVariant(const FString& PedId, FName VariantId);
	bool ExecSpawn(const FString& PedId, const FVector& Loc, float YawDeg, FName VariantId, bool bUseProvidedGroundPoint = false);
	bool ExecRelease(const FString& PedId);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Crowd")
	FCrowdSpawnResult SpawnCrowd(const FCrowdSpawnRequest& Request);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Crowd")
	bool ClearCrowdGroup(FName GroupId);

	UFUNCTION(BlueprintCallable, Category = "Pedestrian|Crowd")
	FCrowdSpawnResult RespawnCrowd(FName GroupId, int32 NewSeed);

private:
	APedestrianCharacter* ResolvePedestrianOrLog(const FString& PedId, const TCHAR* CommandName) const;
	bool ResolveCrowdAssets(const FCrowdSpawnRequest& InRequest, FCrowdSpawnRequest& OutResolvedRequest, FString& OutError) const;
	bool BuildCandidateSelections(
		const UCrowdAppearancePool* AppearancePool,
		const UCrowdRoleProfile* RoleProfile,
		TArray<FCrowdRuntimeSelection>& OutCandidates,
		FString& OutError) const;
	bool SelectAppearance(const TArray<FCrowdRuntimeSelection>& Candidates, FRandomStream& Stream, FCrowdAppearanceEntry& OutAppearance) const;
	bool SelectSpawnLocation(
		const FCrowdSpawnRequest& Request,
		const UCrowdRoleProfile* RoleProfile,
		const TArray<FVector>& ExistingLocations,
		FRandomStream& Stream,
		FVector& OutLocation) const;
	void GatherSelectedAccessories(const FCrowdAppearanceEntry& Appearance, FRandomStream& Stream, TArray<FCrowdAccessorySpec>& OutAccessories) const;
	bool ClearCrowdGroupInternal(FName GroupId, bool bKeepGroupState);

	TMap<FString, TWeakObjectPtr<APedestrianCharacter>> PedMap;
	TSet<FString> DynamicPedIds;
	TMap<FName, FCrowdGroupState> CrowdGroups;
};
