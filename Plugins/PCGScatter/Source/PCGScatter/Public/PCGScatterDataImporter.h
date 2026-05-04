// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "ogr_geometry.h"
#include "PCGScatterCommon.h"
#include "Subsystems/WorldSubsystem.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "SUbsystems/EngineSubsystem.h"
#include "PCGScatterDataImporter.generated.h"

struct FPCGScatterData;
class OGRPoint;

struct ArrayOfPCGScatterData
{
	TArray<FPCGScatterDataSavInfo> PCGScatterDataSavInfos;
};

USTRUCT(BlueprintType)
struct FPCGScatterIntersectedPolyList
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterDataImporter")
	TArray<int32> IndexList;

	FPCGScatterIntersectedPolyList() {};

	FPCGScatterIntersectedPolyList(const TArray<int32>& InIndexList)
		: IndexList(InIndexList) {}
};

USTRUCT(BlueprintType)
struct FPCGScatterGroundContainsPolyInfo
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterDataImporter")
	TArray<FPCGScatterIntersectedPolyList> ArrayOfIndexList;
};

UCLASS()
class PCGSCATTER_API UPCGScatterDataImporter : public UEngineSubsystem
{
public:
	GENERATED_BODY()

	static int32 GetCityVCoordinateSysEnum_EPSGInterger();

	/** 获取 CityV 项目当前的中心点，注意要与 GetCachedOffsetCenter 有所区分 */
	static OGRPoint GetCityVProjectOffsetCenter();

	/** 获取该 Impoter 缓存的一个项目中心点，多用于编辑器下的数据处理 */
	static OGRPoint GetCachedOffsetCenter();

	static OGRPoint GetAssignedSpatialReferenceOffsetCenter();

	static void ResetCachedOffsetCenter();

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	void SetOffsetCenter(const FString& Longitude, const FString& Latitude);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	void SetOffsetCenterDouble(double Longitude, double Latitude);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	void SetGameSettingOffsetCenter(const FVector& Center);

	static void BuildOGRCoordinateTransformation(
		OGRCoordinateTransformation*& OutTransform,
		int32 InSourceSRS,
		int32 InTargetSRS);

	static void BuildOGRCoordinateTransformation(
		OGRCoordinateTransformation*& OutTransform,
		OGRSpatialReference& pSourceSRS,
		OGRSpatialReference& pTargetSRS,
		int32 InSourceSRS,
		int32 InTargetSRS);
	/**
	 * 做投影变换，需要先变换后再减去偏移量
	 * @param pPointToOffset 原始点
	 * @param pProjectCenter 项目中心点。这里传入的参数需要保证是设置过源坐标系，即需要经过 pProjectCenter.assignSpatialReference(&SourceSRS)
	 */
	static void ApplyPointOffset(OGRPoint& pPointToOffset, OGRPoint pProjectCenter, OGRCoordinateTransformation* InTransformation = nullptr);

	static void ApplyPointOffset(OGRPoint& pPoint, int32 InSourceSRS, int32 InTargetSRS);

	static void ApplyOGRGeometryOffset(
		OGRGeometry& pGeometry,
		const OGRwkbGeometryType& InGeometryType,
		OGRCoordinateTransformation* pTransformation,
		OGRPoint CenterPoint_AssignedSpatialReference);

	static void ApplyOGRGeometryOffset(
		OGRGeometry& pGeometry,
		const OGRwkbGeometryType& InGeometryType,
		int32 InSourceSRS,
		int32 InTargetSRS);


	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	void ImportGeoJson_Point(const FString& FilePath, FPCGScatterData& PointData);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	void ImportGeoJson_LineString(const FString& FilePath, FPCGScatterData& LineData);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	void ImportGeoJson_LineString_CustomProj(const FString& FilePath, FPCGScatterData& LineData);

	static class GDALDataset* ParseOGRGeometryInternal_TransformedSRS(
		const FString& FilePath,
		TArray<OGRGeometry*>& OutGeometries,
		TMap<int32, FPCGScatterGeometryProperty>& OutProperties,
		int32 InSourceSRS,
		int32 InTargetSRS);

	static void GDALPointsToPCGScatterPoints(
		TArray<OGRGeometry*>& InGeometries,
		const OGRwkbGeometryType& InGeometryType,
		TArray<TArray<FSplinePoint>>& OutGeometries_UEFormat);

	static void GDALDataToPCGScatterData(
		TArray<OGRGeometry*>& InGeometries,
		const OGRwkbGeometryType& InGeometryType,
		FPCGScatterData& OutPCGScatterData,
		TMap<int32, FPCGScatterGeometryProperty>& OutProperties
		);

	static void QueryIntersectedPolygons(
		TArray<OGRGeometry*>& InGeometries,
		TMap<int32, TArray<int32>>& Out_PolyIdx_To_OtherIntersectedPolyIndices
		);

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "PCGScatterDataImporter")
	void ImportGeoJson_Polygon(
		const FString& FilePath,
		FPCGScatterData& PolygonData,
		int32 InSourceSRS = 4326,
		int32 InTargetSRS = 3857
		);

	/**
	 *
	 * 相邻建筑群。表中存储的结构为 [(0,2,4),(1),(3,5),...]，同一个Item中存储的是相邻的建筑 (0,2,4)。
	 * 建筑所属的地块。表中存储的结构为 [{0 : [(0,2,4)]}, {1 : [(1), (3,5)]}]，同一个Item中存储的是 {地块编号：地块上的建筑群} 的映射。
	 */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "PCGScatterDataImporter")
	void ImportGeoJson_Polygon_ClusteredBuildings(
		const FString& InBuildingFilePath,
		const FString& InGroundSectionFilePath,
		FPCGScatterData& OutBuildingData,
		FPCGScatterData& OutGroundData,
		FPCGScatterData& OuterContour,
		TArray<FPCGScatterIntersectedPolyList>& OutIntersectedPolygonsTable,
		TMap<int32, FPCGScatterGroundContainsPolyInfo>& Out_Ground_To_Buildings_Table,
		const int32 InSourceSRS = 4326,
		const int32 InTargetSRS = 3857
		);

	static void PostProcess_MappingGroundSection2Buildings(
		TArray<OGRGeometry*>& OutBuildings,
		const TArray<OGRGeometry*>& InGroundSections,
		FPCGScatterData& OutBuildingData,
		FPCGScatterData& InGroundSectionData,
		TMap<int32, FPCGScatterGroundContainsPolyInfo>& OutGround2PolygonsTable
		);

	static void MergeConnectedComponents(
		const TMap<int32, TArray<int32>>& IndexMap,
		TArray<FPCGScatterIntersectedPolyList>& OutConnectedComponents);

	static void PostProcess_CalculateOuterContour(
		const TArray<FPCGScatterIntersectedPolyList>& OutIntersectedPolygonsTable,
		const TArray<OGRGeometry*>& InBuildings,
		FPCGScatterData& OutBuildingData,
		FPCGScatterData& OuterContour
		);


	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	void ImportGeoJson_Polygon_CustomProj(const FString& FilePath, FPCGScatterData& PolygonData);

	// 建筑生成专用函数，借助OGRPolygon->getCentroid() 额外写入一个重心坐标的Property,供后续使用
	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	void ImportGeoJson_Polygon_ForBuilding(const FString& FilePath, FPCGScatterData& PolygonData);

	UFUNCTION(BlueprintCallable, meta = (WorldContext = "WorldContextObject"), Category = "PCGScatterDataImporter")
	void ImportPCGScatterData(UObject* WorldContextObject, const FString& FilePath, TArray<FGuid>& NewAddedPCGScatterData);

	void ParsePCGScatterData(UObject* WorldContextObject, const FPCGScatterDataSavInfo& InPCGScatterDataSavInfo);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterDataImporter")
	bool IsPCGScatterDataFile(const FString& FilePath);

private:
	static FVector _CoordinateCenter;
};