// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "DtfGeomStructsDefine.h"
#include "UDynamicMesh.h"
#include "Components/SplineComponent.h"
#include "Components/Widget.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UObject/ObjectMacros.h"
#include "DynamicBuildingGeoFunctions.generated.h"

UCLASS()
class DYNAMICBUILDING_API UDynamicBuildingGeoFunctions : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Geo", meta = (WorldContext="WorldContextObject"))
	void Get2DBillBoardSizeIn3D(const FVector& In3DLocation, const double In2DSize, const FVector& InCameraLocation, APlayerController* InPlayerController, UObject* WorldContextObject, double& Out3DSize, bool& OutValid);

	// 在 X0Y 平面上计算
	static bool FindRayIntersection(
		const FVector& Start1, const FVector& Direction1,
		const FVector& Start2, const FVector& Direction2,
		FVector& IntersectionPoint
		);
	/**
	 * 对于一组逆时针缠绕的点位，沿着顶点 RightVector 方向，偏移 abs(InDistance) 的距离；若 InDistance 为负值，则是向 LeftVector 方向偏移 abs(InDistance) 距离。
	 * @param InOriPts 原始需要偏移的坐标点
	 * @param InDistance 偏移的距离，有正负值之分
	 * @param InClosed 原始的一组点是否有首尾重合的点
	 * @param WorldContextObject 
	 * @param OutPts 
	 */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Geo", meta = (WorldContext="WorldContextObject"))
	static void OffsetPolylines_CPP(const TArray<FVector>& InOriPts, const double InDistance, const bool InClosed, TArray<FVector>& OutPts);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Widget", meta = (WorldContext="WorldContextObject"))
	void WidgetContainsOthersInSameCanvas(UWidget* InContainer, UWidget* InTarget, UObject* WorldContextObject, bool& OutContains);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void GetBoundOfVector2DArray_CPP(const TArray<FVector2D>& InVector2D, FVector2D& OutRangeX, FVector2D& OutRnageY);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default", meta = (WorldContext="WorldContextObject"))
	static void DivideSplineByCount_CPP(USplineComponent* InSpline, const int32 InCountNum, const ESplineCoordinateSpace::Type InCoordinateSpace, const bool bUseScale, UObject* WorldContextObject, TArray<FTransform>& OutTransformArray, TArray<FVector>& OutLocationArray, TArray<FVector2D>& OutVector2DArray, TArray<double>& OutTValueArray);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default", meta = (WorldContext="WorldContextObject"))
	static void ConvertTransArrayToVec2DArray_CPP(UPARAM(ref) TArray<FTransform>& InTransform, UObject* WorldContextObject, TArray<FVector2D>& OutVec2D);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Polyline", meta = (WorldContext="WorldContextObject"))
	static void ClosestPointToPolyline_CPP(const FDtfGeomPolyline& InPolyline, const FVector& InTestPoint, FVector& OutClosestPt, double& OutMinDistance, double& OutTValue, bool& OutValid);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default", meta = (WorldContext="WorldContextObject"))
	static void ClosestPointToLineSegment_CPP(const FVector& LineStart, const FVector& LineEnd, const FVector& InTestPoint, const bool bCalculateDistance, FVector& OutClosestPoint, double& OutTValue, double& OutDistance);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default", meta = (WorldContext="WorldContextObject"))
	static void NumberSequence_Int_CPP(const int32 InStartNum, const int32 InCount, const int32 InStep, UObject* WorldContextObject, TArray<int32>& OutSeq);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default")
	static void GetSplinePoints_CPP(const USplineComponent* InSpline, const ESplineCoordinateSpace::Type InCoordinateSpace, bool bUseScale, TArray<FTransform>& OutTransArray, TArray<FVector>& OutVectorArray, TArray<FVector2D>& OutVector2DArray);

	static void DivideLineFrom2EndsByLength(const FDtfGeomLineSegment& InLineSegment,const FVector2D& InDivideLengthAt2Ends, const double InMinMidLength, FDtfGeomLineSegment& OutFirstLineSegm, FDtfGeomLineSegment& OutSecondLineSegm, FDtfGeomLineSegment& OutThirdLineSegm, bool& OutSuccess);

	static void DivideLineFrom2EndsByRatio(const FDtfGeomLineSegment& InLineSegment,const FVector2D& InDivideRatioAt2Ends, const double InMinMidLength, FDtfGeomLineSegment& OutFirstLineSegm, FDtfGeomLineSegment& OutSecondLineSegm, FDtfGeomLineSegment& OutThirdLineSegm, bool& OutSuccess);
	
	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void DivideLineWith2Ends_CPP(const FDtfGeomLineSegment& InLineSegment, const FVector2D& InDivideRatioAt2Ends, const FVector2D& InDivideLengthAt2Ends, const bool bIsByRate, const double InMinMidLength, FDtfGeomLineSegment& OutFirstLineSegm, FDtfGeomLineSegment& OutSecondLineSegm, FDtfGeomLineSegment& OutThirdLineSegm, bool& OutSuccess);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default")
	static void LineDirection_CPP(const FDtfGeomLineSegment& InLine, FVector& OutDir);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default")
	static void VectorArrayToVector2DArray_CPP(const TArray<FVector>& InV, TArray<FVector2D>& OutV2D);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default", meta = (WorldContext="WorldContextObject"))
	static void GenerateWall_CPP(const FDtfGeomPolyline& InPolyline, UPARAM(ref) UDynamicMesh*& InDynamicMesh, const double InOffsetDistance, const double InHeight, const float InExtrudeHeight, const float InOptionsUVScale, UDynamicMesh*& OutTargetMesh);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Polyline")
	static void GetPolylineLength_CPP(const FDtfGeomPolyline& InPolyline, double& OutLen, bool& OutValid);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default")
	static void DivideLineByLength_CPP(const FDtfGeomLineSegment& InLineSegment, const double InLength, TArray<FVector>& OutPoints, TArray<double>& OutTValue);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Polyline")
	static void GetPolylinePointsTransform_CPP(const FDtfGeomPolyline& InPolyline, const bool bIsClosed, TArray<FTransform>& OutTrans);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default")
	static void DivideDomain_CPP(const FVector2D& InDomain, const int32 InCount, TArray<float>& OutSeq);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void SplinePointsToPolyline_CPP(const TArray<FSplinePoint>& InSplinePts, const bool bSetOriPt, const FVector& InOriPt, FDtfGeomPolyline& OutPolyline);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void GetPolylineDirectionAtSegment_CPP();

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Polyline")
	static void GetPolylineDirectionAtPt_CPP(const FDtfGeomPolyline& InPolyline, const int32 Index, FVector& OutDir, bool& OutSuccess);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Polyline")
	static void GetPolylineHorizontalRightDirectionAtPt_CPP(const FDtfGeomPolyline& InPolyline, const int32 Index, FVector& OutRight, bool& OutSuccess);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void ShiftVectorArray_CPP(const TArray<FVector>& InShiftVectorArray, const int32 Count, TArray<FVector>& OutVectorArray);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void DividePolylineByLength_CPP(const FDtfGeomPolyline& InPolyline, const double Length, TArray<FVector>& OutDividePoint, TArray<double>& OutTValue, TMap<int32, FDtfGeomPolyline>& OutInterger2PolylineMap);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void GetPolylineSegments_CPP(const FDtfGeomPolyline& InPolyline, const bool bComputeLength, TArray<FDtfGeomLineSegment>& OutLineSegments, TArray<double>& OutLinesLength);

	/**
	 * 对给定的 Polyline 采样 TValue 处的点位和切线。TValue 分为两种情况，一种是 Normalized, 取值范围在[0,1];另一种是 "[点数-1].当前段百分比" 形式。
	 * @param InPolyline 
	 * @param bNormalizedTValue 
	 * @param InTValue 
	 * @param OutPt 
	 * @param OutTanget 
	 * @param OutValid 
	 */
	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void SamplePolyline_CPP(const FDtfGeomPolyline& InPolyline, const bool bNormalizedTValue, const double InTValue, FVector& OutPt, FVector& OutTanget, bool& OutValid);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void ExtendPolyline_CPP(const FVector2D& InExtend, const FDtfGeomPolyline& InPolyline, const bool bExtendSingleSegment, FDtfGeomPolyline& OutDtfGeomPolyline);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void OffsetDTFPolyline_CPP(const FDtfGeomPolyline& InPolyline, const double InDistance, const bool bClosed, FDtfGeomPolyline& OutDtfGeomPolyline);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Polyline")
	static void GetLineLength_CPP(const FDtfGeomLineSegment& InLineSegment, double& OutLength);

	UFUNCTION(BlueprintCallable, BlueprintPure, Category="DynamicBuildingGeoFunctions | Default")
	static void LineOffset_CPP(const FDtfGeomLineSegment& InLineSegment, const double Distance, FDtfGeomLineSegment& OutLineSegment);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void ProjectVector2DToLine2D_CPP(const FVector2D& InTestPoint, const FVector2D& InTestDir, const FDtfGeomLineSegment& InLineSegment, bool& OutValid, FVector2D& OutProjPt, double& OutDistance);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void ProjectVector2DToPolyline2D_CPP(const FVector2D& InTestPoint, const FVector2D& InTestDir, const FDtfGeomPolyline& InPolyline, bool& OutValid, FVector2D& OutProjPt, double& OutDis);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void CutPolyline_CPP(const FVector2D& Pt, const FVector2D& Dir, const FDtfGeomPolyline& InPolyline, FDtfGeomPolyline& OutL1, FDtfGeomPolyline& OutL2, FVector& OutCutPt1, FVector& OutCutPt2);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void BreakPolylineWithAngle_CPP(const FDtfGeomPolyline& InPolyline, const double InMin, const double InMax, TArray<FDtfGeomPolyline>& OutPolylineArray, TArray<FVector2D>& OutTRange1);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | DMC")
	static void GetDMCTriangles_CPP(UDynamicMesh* InDynamicMesh, TArray<FIntVector>& OutTriangles);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | DMC")
	static void GetDMCUVs_CPP(UDynamicMesh* Mesh, const int32 UVChannel, TArray<FIntVector>& OutTriangles);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void VectorToHorizontalRotation_CPP(const FVector& InVector, const double InAddition, FRotator& OutRotator);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void Fillet_CPP(const FDtfGeomPolyline& InPolyline, const double InR, const double DivideStep, TArray<FVector>& OutVC);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void ArcByCenter_CPP(const FVector& InCenter, const FVector& In1stPt, const FVector& In2ndPt, TArray<FVector>& OutArc);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static bool IsPolylineClosed_CPP(const FDtfGeomPolyline& InPolyline, const double Tolerance);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void TransformArrayWorldToRelative_CPP(const TArray<FTransform>& TransformArrayWorld, const FTransform& InParentTransform, TArray<FTransform>& OutTrans);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void VectorArrayWorldToRelative_CPP(const TArray<FVector>& VectorArrayWorld, const FTransform& InParentTransform, TArray<FVector>& OutVec);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Polyline")
	static void AppendExtrudedFacadeFromPolyline_CPP(UDynamicMesh* DynamicMesh, const double Height, const int32 InMatID, const FDtfGeomPolyline& InPolyline);

	UFUNCTION(BlueprintCallable, Category="DynamicBuildingGeoFunctions | Default")
	static void ForcePolylineFlattenByHeight_CPP(const FDtfGeomPolyline& InPolyline, const double Height, FDtfGeomPolyline& OutDtfGeomPolyline);
};