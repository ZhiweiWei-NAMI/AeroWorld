// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "ArchitectureStructs/CommonStructsDefine.h"
#include "PCGScatterCommon.h"
#include "GameFramework/Actor.h"
#include "CPP_ArchitectureManager.generated.h"

class UPCGScatterPointData;

/** Please add a class description */
UCLASS(Blueprintable, BlueprintType)
class ACPP_ArchitectureManager : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ACPP_ArchitectureManager();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;
public:
	
public:
	/** Please add a variable description */
	//UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	//TObjectPtr<class UBPC_Archi_LightManager_C> BPC_Archi_LightManager;

	/** Please add a variable description */
	//UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	//TObjectPtr<class UBPC_Archi_PlantsManager_C> BPC_Archi_PlantsManager;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<USceneComponent> DefaultSceneRoot;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	TArray<FVector> VectorArray1;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	TObjectPtr<UCurveLinearColor> ColorCrv;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default")
	TMap<EDynamicBuildingLanduse ,double> PodiumBuildingHeightMap;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	TMap<EDynamicBuildingLanduse ,double> LevelHeightMap;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	TArray<UMaterialInterface*> RoofMaterial;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	int32 Seed;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default")
	int32 Min;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default")
	int32 Max;
};

