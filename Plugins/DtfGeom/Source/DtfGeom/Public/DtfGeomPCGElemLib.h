// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "PCGElement.h"
#include "PCGSettings.h"
#include "PCGPin.h"
#include "DtfGeomPCGElemLib.generated.h"

struct FGeometryScriptSpatialQueryOptions;
class UDynamicMesh;
/**
 * 
 */
UCLASS()
class DTFGEOM_API UDtfGeomPCGElemLib : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static void DmcSample(UDynamicMesh* TargetMesh, UPARAM(ref) FPCGContext& InContext, const FPCGDataCollection& InputCollection, FPCGDataCollection& OutputCollection);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static void ProjectPointsToDmcMesh(
		UDynamicMesh* TargetMesh,
		const TArray<UDynamicMesh*> ExcludedMeshes,
		FVector ProjectionTargetBoundsTopPlane,
		FVector ProjectionDirection,
		FGeometryScriptSpatialQueryOptions Options,
		UPARAM(ref) FPCGContext& InContext,
		const FPCGDataCollection& InputCollection,
		FPCGDataCollection& OutputCollection,
		bool bDestroyTargetMeshAffterProjection=true
		);

	UFUNCTION(BlueprintCallable, Category = "DtfGeom")
	static void FilterPointsOutOfCoutour(
		UPARAM(ref) FPCGContext& InContext,
		const FPCGDataCollection& InputCollection,
		FPCGDataCollection& OutputCollection,
		const TArray<FVector>& OuterContourPoints
		);
};


/** Filters a data collection based on some tag criterion */
UCLASS(BlueprintType, ClassGroup = (Procedural))
class DTFGEOM_API UPCGPartitionByTagSettings : public UPCGSettings
{
	GENERATED_BODY()

public:
	//~Begin UPCGSettings interface
#if WITH_EDITOR
	virtual FName GetDefaultNodeName() const override;
	virtual FText GetDefaultNodeTitle() const override;
	virtual FText GetNodeTooltipText() const override;
	virtual EPCGSettingsType GetType() const override { return EPCGSettingsType::Filter; }
#endif


protected:
	virtual TArray<FPCGPinProperties> OutputPinProperties() const override;
	virtual FPCGElementPtr CreateElement() const override;
	//~End UPCGSettings interface

public:
	/** Comma-separated list of tags */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = Settings, meta = (PCG_Overridable))
	TArray<FString> SelectedTags;
};

class FPCGPartitionByTagElement : public FSimplePCGElement
{
protected:
	virtual bool ExecuteInternal(FPCGContext* Context) const override;
};