// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"
#include "SceneDBManager.generated.h"

struct FArchiGeneratingInfo;
/**
 * 
 */
UCLASS()
class DYNAMICBUILDING_API USceneDBManager : public UEngineSubsystem
{
	GENERATED_BODY()

public:
	/** 渐进式生成设计好的多种建筑 */
	UFUNCTION(BlueprintCallable, meta = (WorldContext = "WorldContextObject"), Category = "DynamicBuilding")
	void ProgressiveGeneration(UObject* WorldContextObject, UPARAM(ref)TArray<FArchiGeneratingInfo>& ArchiGeneratingInfos, int32 BatchProcessCount = 20);

	UFUNCTION(BlueprintCallable, meta = (WorldContext = "WorldContextObject"), Category = "DynamicBuilding")
	void AllTotalGenerate();

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "DynamicBuilding")
	TArray<AActor*> CachedArchiObject;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "DynamicBuilding")
	int32 CurrentProcessIndex = 0;

	UPROPERTY()
	FTimerHandle TimerHandle;

};