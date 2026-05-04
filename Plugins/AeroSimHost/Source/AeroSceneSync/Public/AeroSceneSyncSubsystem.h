#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroSceneSyncSubsystem.generated.h"

class FJsonObject;
class FJsonValue;

UCLASS()
class AEROSCENESYNC_API UAeroSceneSyncSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;

	TSharedPtr<FJsonObject> ApplyFrame(const TSharedPtr<FJsonObject>& Payload, FString& OutError);
	void ResetSyncState();

private:
	bool ApplySpawnDelta(const TSharedPtr<FJsonObject>& DeltaObject, TArray<TSharedPtr<FJsonValue>>& OutSpawnResults, FString& OutError);
	bool ApplyUpdateDelta(const TSharedPtr<FJsonObject>& DeltaObject, TArray<TSharedPtr<FJsonValue>>& OutUpdateResults, FString& OutError);
	bool ApplyRemoveDelta(const TSharedPtr<FJsonObject>& DeltaObject, TArray<TSharedPtr<FJsonValue>>& OutRemoveResults, FString& OutError);
	bool ReadPoseField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutPositionEnuM, FRotator& OutRotationDeg) const;
	void ReadTagsField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, TArray<FString>& OutTags) const;

private:
	TMap<FString, FString> EntityToProxyInstance;
};
