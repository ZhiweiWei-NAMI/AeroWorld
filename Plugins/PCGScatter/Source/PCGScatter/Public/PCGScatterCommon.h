// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "JsonObjectConverter.h"
#include "Elements/PCGSplineSampler.h"
#include "Components/SplineComponent.h"
#include "Engine/StaticMesh.h"
#include "UObject/SoftObjectPtr.h"
#include "PCGScatterCommon.generated.h"

UENUM(BlueprintType)
enum class EPCGScatterType : uint8
{
	PCGScatterType_Unknow,
	PCGScatterType_Point,
	PCGScatterType_Line,
	PCGScatterType_Polygon
};

/** 智能布局-点、线、面布局所需的配置项 */
USTRUCT(BlueprintType)
struct FPCGScatterStrategy
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	EPCGScatterType PCGScatterType = EPCGScatterType::PCGScatterType_Point;

	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	FName SourceSplineTag;

	/** bSpawnStaticMeshOrActor=0，生成StaticMesh; bSpawnStaticMeshOrActor=1, 生成Actor */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	bool bScatterActor = false;

	/** 程序化生成时，采样Spline需要设置的参数 */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	FPCGSplineSamplerParams PCGSplineSamplerParams;

	/** 程序化生成StaticMesh需要设置的参数 */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	TMap<TSoftObjectPtr<UStaticMesh>, int> StaticMeshs;

	/** 程序化生成Actor需要设置的参数 */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	TSubclassOf<AActor> TemplateActorClass = nullptr;

	/** 程序化生成Actor需要设置的参数 */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	TArray<FName> TagsToAddOnActor;

	/** 程序化生成的结果是否需要进行投影到物体上; 如果需要投影，设置要投影到的Actor的标签ProjectTargetTag */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	FString ProjectTargetTag;

	/** 程序化生成的物体的Transform */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	FTransform TransformStart;

	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	FTransform TransformEnd;

	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "PCGScatterCommon")
	int Seed = 0;
};

USTRUCT(BlueprintType, Blueprintable)
struct FPCGScatterGeometryProperty
{
	GENERATED_BODY()
	/** 几何对象存储的属性对
	 * key: 属性名
	 * value: 属性值
	 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	TMap<FString, FString> PropertyMap;
};

UCLASS(BlueprintType)
class PCGSCATTER_API UPCGScatterGeometryData : public UObject
{
	GENERATED_BODY()

public:
	UPCGScatterGeometryData() {}

	UPCGScatterGeometryData(EPCGScatterType InType)
		: Type(InType) {}

	virtual EPCGScatterType GetDataType() const { return Type; }

	virtual void RemoveAllDataFromRoot(){};

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	EPCGScatterType Type = EPCGScatterType::PCGScatterType_Unknow;
};

/** MultiPoint 点数组对象。CityV中默认使用一组点，个数[0,n-1] */
UCLASS(BlueprintType)
class PCGSCATTER_API UPCGScatterPointData : public UPCGScatterGeometryData
{
	GENERATED_BODY()

public:
	UPCGScatterPointData();

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void SetPoints(UPARAM(ref)TArray<FSplinePoint> InPoints);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	TArray<FSplinePoint> GetPoints();

	TArray<FSplinePoint>& GetPointsMutable();

	virtual void RemoveAllDataFromRoot() override;

private:
	// 标记为 UPROPERTY 的属性在使用 DuplicateObject 时会被拷贝
	UPROPERTY()
	TArray<FSplinePoint> _Points;
};

/** MultiLine 线数组对象 */
UCLASS(BlueprintType)
class PCGSCATTER_API UPCGScatterLineData : public UPCGScatterGeometryData
{
	GENERATED_BODY()

public:
	UPCGScatterLineData()
		: UPCGScatterGeometryData(EPCGScatterType::PCGScatterType_Line)
	{
	}

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void SetLines(const TArray<UPCGScatterPointData*> InLines);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void AddLine(UPCGScatterPointData* Line);

	/** UI从列表删除某条线 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void DeleteLine(const int32 LineIndex);

	/** UI从列表更新某条线 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void UpdateLine(const int32 LineIndex, UPCGScatterPointData* NewLine);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void UpdateLineFromSplinePoint(const int32 LineIndex, UPARAM(ref) TArray<FSplinePoint>& NewSplinePoints);
	
	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	TArray<UPCGScatterPointData*> GetLines();

	virtual void RemoveAllDataFromRoot() override;

private:
	UPROPERTY()
	TArray<UPCGScatterPointData*> _Lines;
};

/** MultiPolygon 面数组对象 */
UCLASS(BlueprintType)
class PCGSCATTER_API UPCGScatterPolygonData : public UPCGScatterGeometryData
{
	GENERATED_BODY()

public:
	UPCGScatterPolygonData()
		: UPCGScatterGeometryData(EPCGScatterType::PCGScatterType_Polygon)
	{
	}

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void SetPolygons(const TArray<UPCGScatterPointData*> InPolygons);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	TArray<UPCGScatterPointData*> GetPolygons();

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void AddPolygon(UPCGScatterPointData* Polygon);

	/** UI从列表删除某个面 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void DeletePolygon(const int32 PolygonIndex);

	/** UI从列表更新某个面 */
	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void UpdatePolygon(const int32 PolygonIndex, UPCGScatterPointData* NewPolygon);

	UFUNCTION(BlueprintCallable, Category = "PCGScatterCommon")
	void UpdatePolygonFromSplinePoints(const int32 PolygonIndex, UPARAM(ref) TArray<FSplinePoint>& NewSplinePoints);
	
	virtual void RemoveAllDataFromRoot() override;

private:
	UPROPERTY()
	TArray<UPCGScatterPointData*> _Polygons;
};

USTRUCT(BlueprintType, Blueprintable)
struct FPCGScatterData
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	FString UIName;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "PCGScatterCommon")
	FGuid GeometryGuid;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "PCGScatterCommon")
	FGuid ParentCollectionGuid;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "PCGScatterCommon")
	FString ParentCollectionUIName;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "PCGScatterCommon")
	FGuid ParentGroupGuid;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "PCGScatterCommon")
	FString ParentGroupUIName;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	UPCGScatterGeometryData* GeometryData;

	/** 集合对象的属性集
	 *	key: 数据对象在GeometryData内元素的序号
	 *	value: 元素对应的属性集，内部是一个Map
	 *	将实体要素的几何数据和属性数据分离，是为了避免Line, Polygon在使用Point构建自己时，出现属性集嵌套的问题。
	 *	**会有MultiPoint、MultiLine、MultiPolygon，或是交互绘制的数据信息，部分数据没有或不需要属性信息的情况，使用TMap方便数据与属性对照起来。
	 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	TMap<int32, FPCGScatterGeometryProperty> Properties;

	FPCGScatterData()
		: GeometryGuid(FGuid::NewGuid()) {}
};

USTRUCT(BlueprintType, Blueprintable)
struct FPCGScatterDataCollection
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	TArray<FPCGScatterData> PCGScatterData;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	FGuid CollectionGuid;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	FGuid ParentGroupGuid;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	EPCGScatterType CollectionType = EPCGScatterType::PCGScatterType_Unknow;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	FString UIName;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	TArray<FGuid> GeometryElems;
};

USTRUCT(BlueprintType)
struct FPCGScatterDataGroup
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	FGuid GroupGuid;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	FString GroupCategory;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	FString UIName;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "PCGScatterCommon")
	// TArray<FPCGScatterCollection> Collections;
	TArray<FGuid> Collections;

	FPCGScatterDataGroup()
		: GroupGuid(FGuid::NewGuid()) {}

	FPCGScatterDataGroup(const FGuid& InGroupGuid)
		: GroupGuid(InGroupGuid) {}
};

/** 只用于数据的存储、恢复使用; C++其它模块及蓝图不能使用 */
USTRUCT()
struct FPCGScatterSavPoints
{
	GENERATED_BODY()

	UPROPERTY()
	TArray<FSplinePoint> Data;
};
/** 只用于数据的存储、恢复使用; C++其它模块及蓝图不能使用 */
USTRUCT()
struct FPCGScatterDataSavInfo
{
	GENERATED_BODY()

	/** 本条数据在大纲上的UIName */
	UPROPERTY()
	FString UIName;

	UPROPERTY()
	EPCGScatterType Type;

	UPROPERTY()
	FGuid GeometryGuid;

	UPROPERTY()
	FGuid ParentCollectionGuid;
	/** 本条数据在大纲上所处的Collection的UIName */
	UPROPERTY()
	FString ParentCollectionUIName;

	UPROPERTY()
	FGuid ParentGroupGuid;
	/** 本条数据在大纲上所处的Collection，Collection所处的GroupUIName */
	UPROPERTY()
	FString ParentGroupUIName;

	/**
	 * 点数据：只有GeometryData[0] 有效
	 * 线数据：GeometryData[n] 包含 n 条线
	 * 面数据：GeometryData[n] 包含 n 个面
	 */
	UPROPERTY()
	TArray<FPCGScatterSavPoints> GeometryData;

	/** 集合对象的属性集
	 *	key: 数据对象在GeometryData内元素的序号
	 *	value: 元素对应的属性集，内部是一个Map
	 *	将实体要素的几何数据和属性数据分离，是为了避免Line, Polygon在使用Point构建自己时，出现属性集嵌套的问题。
	 */
	UPROPERTY()
	TMap<int32, FPCGScatterGeometryProperty> Properties;

};
