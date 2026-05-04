// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "DtfGeomStructsDefine.h"
#include "DynamicMeshActor.h"
#include "PCGScatterCommon.h"
#include "ArchitectureStructs/CommonStructsDefine.h"
#include "GeometryScript/MeshSamplingFunctions.h"
#include "CPP_ArchitectureParent.generated.h"

/**
 * 
 */

enum class EDynamicBuildingLanduse : uint8;
class USplineComponent;

UCLASS(Blueprintable, BlueprintType)
class ACPP_ArchitectureParent : public ADynamicMeshActor
{
	GENERATED_BODY()

public:
	ACPP_ArchitectureParent();

protected:
	virtual void BeginDestroy() override;

public:
	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateWallBySweep(const FDtfGeomPolyline& InPolyline, FVector2D InOffsets, bool bClosed, double InHeight, UDynamicMesh*& OutTargetMesh);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateWallByOffset(const FDtfGeomPolyline& InPolyline, bool bClosed, const FVector2D& InOffsets, double InHeight, const FTransform&
		InTransform, UDynamicMesh*& OutTargetMesh);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateFinOnLine(const FDtfGeomLineSegment& InLineSegment, const double InLength, const double InWidth, const double InThickNess, const
		double InHeight, const bool bIncludeEnds, UDynamicMesh*& OutTargetMesh);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void DividePolylineByCorner(const FDtfGeomPolyline& InPolyline, const double InCornerWidth, const double InMinMidLength, TArray<FDtfGeomPolyline>& OutCornerPolyline, TArray<FDtfGeomLineSegment>& OutMiddlePolyline, bool& OutSuccess);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void MyAppendMesh(UDynamicMesh* AppendMesh, FTransform AppendTransform, bool bReset, bool bReleaseAll);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void FloorPanel(double InOffset, const FDtfGeomPolyline& InOutline, double InHeight, int32 InMaterialID, bool InFlipOrientation, double InUVScaleFactor, bool InbSample, FGeometryScriptMeshPointSamplingOptions InSampleOption, FLinearColor InColor, TArray<FTransform>& OutSamples);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateFinGroupOnLine(const FDtfGeomLineSegment& InLineSegment, FDynamicBuildingFinGroupStyle InFinGroupStyle, bool bIncludeEnds, double InDivideLength, double InHeight, double InStartHeight, UDynamicMesh*& OutTargetMesh);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateFacades(const TArray<FDtfGeomLineSegment>& InLineSegments, const FVector2D& InTexUVCount, double InDivideLength, double InHeight, int32 InLevelCount, double InBottomHeight, int32 InToMatID, FLinearColor InColor, UDynamicMeshComponent* InCustomDMC, bool bReverseU);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateHoleFrame(const FDtfGeomPolyline& InPolyline, const FVector2D& InHeightRange, const double& InPanelThickness, const double&
		InColumnThickNess, const double& InOffset, const FVector2D& InTexUVCount);

	/** Please add a function description */
	UFUNCTION(BlueprintCallable, Category="Data")
	void AcceptData(const FPCGScatterData& Data);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateBulgesOnLine(
		const FDtfGeomLineSegment& InLineSegment,
		const double& InDistance,
		const FVector2D& InSplitRate,
		const double& InHeight,
		const int32& InLevelCount,
		const double& InBottomHeight,
		const TArray<FDynamicBuildingMateriallayout>& InMaterialInfos,
		const double& InDivideLength,
		const bool& bSingleTexU,
		UDynamicMeshComponent* CustomDMC,
		TArray<FDtfGeomLineSegment>& OutLineSegmentsArray,
		FDtfGeomPolyline& OutDtfGeomPolyline);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void ExtrudePolylineUnweld(
		const FDtfGeomPolyline& InPolyline,
		const FVector& InExturdeVector,
		const FVector2D& InURange,
		const FVector2D& InVRange,
		const int32& InMaterialID,
		const FLinearColor& InColor,
		UDynamicMesh*& OutTargetMesh);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateBulgesOnPolyline(FDtfGeomPolyline Polyline, double Distance, FVector2D SplitRate, double Height, int32 LevelCount, double BottomHeight, UPARAM(ref) TArray<FDynamicBuildingMateriallayout>& MaterialInfos, double DivideLength, bool SingleTexU, TArray<FDtfGeomPolyline>& PolylineArray1, FDtfGeomPolyline& DtfGeomPolyline, TArray<FVector> V, TArray<FDtfGeomPolyline> PolylineArray);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateFacadesUnderPitchRoof(UPARAM(ref) TArray<FDtfGeomLineSegment>& Lines, FVector2D TexUVCount, double DivideLength, double Height, int32 LevelCount, double BottomHeight, UPARAM(ref) TArray<int32>& Mat, FLinearColor Color, FVector PlaneOri, FVector PlaneNormal);

	/** Please add a function description */
	UFUNCTION(BlueprintPure, Category = "DynamicBuiilding")
	void ProjectToPitchedRoof(FVector PlaneOrigin, FVector PlaneNormal, FVector InputPin, FVector& Pt, bool& Valid);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void LocalTransformsToWorld(UPARAM(ref) TArray<FTransform>& Trans, bool RandomRotate, bool RandomScale, FVector2D RotateRange, FVector2D XScaleRange, FVector2D YScaleRange, FVector2D ZScaleRange, TArray<FTransform>& T1, TArray<FTransform> T);

	/** Please add a function description */
	UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void LineLayoutInOutline(FDtfGeomPolyline InOutline, double Length, TArray<FTransform>& TransCache1, TArray<FTransform> TransCache);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void LoadMaterialBasic(TMap<int32, FDynamicBuildingMateriaSetup> MaterialSetupInfo);

	/** Please add a function description */
	UFUNCTION(BlueprintCallable, Category="Default")
	void onePitch(const FDtfGeomPolyline& DtfGeomPolyline, FVector PlaneOrigin, FVector Normal, FVector Dir, FLinearColor Color, int32 RoofID, UPARAM(ref) TArray<int32>& RandomWallID, TArray<FVector> VV1, TArray<FVector> NormalCaches);

	/** Please add a function description */
UFUNCTION(BlueprintCallable, Category="DynamicBuiilding")
	void GenerateWallByOffsetWithDifferentTopAndBottom(FDtfGeomPolyline Polyline, bool Closed, FVector2D Offsets, double height, FTransform Transform, int32 WallMaterialID, int32 TopMaterialID, int32 BottomMaterialID, FVector2D URange, FVector2D VRange, FLinearColor Color, TArray<FVector> V1, TArray<FVector> V2, TArray<FVector> V3);

public:
	/** Please add a variable description */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<USplineComponent> OutlineSpline;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default", meta=(ExposeOnSpawn="true"))
	int32 LevelNum;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default")
	double LevelHeight = 420.0;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default", meta=(ExposeOnSpawn="true"))
	int32 TopLevel;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default", meta=(ExposeOnSpawn="true"))
	int32 BottomLevel;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default", meta=(ExposeOnSpawn="true"))
	FDtfGeomPolyline Outline;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default", meta=(ExposeOnSpawn="true"))
	bool bUseData;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	bool NewVar;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default", meta=(ExposeOnSpawn="true"))
	EDynamicBuildingLanduse Landuse;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	double RoofHeight;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default")
	double MyArea;

	/** Please add a variable description */
	//UPROPERTY(BlueprintReadWrite, EditInstanceOnly, Category="Default", meta=(ExposeOnSpawn="true"))
	//TObjectPtr<class ABP_ArchitectureManager_C> MyManager;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default", meta=(ExposeOnSpawn="true"))
	FString Tag;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	TMap<FName, int32> MaterialIDMap;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Default")
	TMap<int32, FDynamicBuildingMateriaSetup> MaterialSetupInfo;
};